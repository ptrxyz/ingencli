#!/usr/bin/env python3
import click
import yaml
import os.path
import numpy as np
import pylab as plt

from ingen.preprocessors import GoogleDatasetProcessor
from ingen.preprocessors import BitbrainsDatasetProcessor
from ingen.preprocessors import UniformDatasetProcessor
from ingen.preprocessors import HotspotsDatasetProcessor
from ingen.preprocessors import DataSourceIO

from ingen.binning import Binning
from ingen.binning import Binning_Types
from ingen.binning import BinningGenerator
from ingen.binning import Pad_Modes

from ingen.histogram import Pad_Values

from ingen.model import Interpolation_Modes
from ingen.model import ModelParams
from ingen.model import Model

from ingen.bundles import BundleGenerator

from ingen.kpis import KPIs

from ingen.plotter import HairyPlotter


def validate_binning_domain(ctx, param, value):
    # domain = None --> return None
    # domain != None --> validate domain: list of floats>0

    if value is None:
        return None

    def positive_float(f):
        if float(f) <= 0:
            raise ValueError(None)
        else:
            return float(f)

    try:
        return [positive_float(x) for x in value.split(",")]
    except ValueError:
        raise click.BadParameter('%s should be a comma-separated list of floats > 0, not \'%s\'' % (param.name, value))


def validate_binning_amount(ctx, param, value):
    def positive_int(i):
        if int(i) <= 0:
            raise ValueError(None)
        else:
            return int(i)

    try:
        if value is None:
            raise ValueError(None)
        if "," in value:
            return [positive_int(x) for x in value.split(",")]
        else:
            return positive_int(value)
    except ValueError:
        raise click.BadParameter('%s should be either an int > 0 or a comma-separated list of ints > 0, not \'%s\'' % (param.name, value))


def validate_binning(ctx, param, value):
    # if filename exists, load from file
    if os.path.isfile(value):
        try:
            with open(value, "r") as f:
                bobj = yaml.load(f, Loader=yaml.UnsafeLoader)
                return Binning.from_dict(bobj)
        except Exception:
            # raise this if file exists but does not contan a binning
            raise click.FileError(value, 'malformed binning file.')
    else:
        # else, treat value as edges
        click.echo("File %s does not exist, treating it as bin edges..." % value)
        try:
            # split string into edges per dimension, then sort and remove duplicates
            edges = [np.unique(np.array(x.split(","), dtype=float))
                     for x in value.split(":")]
            # all edges must be positive
            if any((edges_along_dim < 0).any() for edges_along_dim in edges):
                raise ValueError(None)
            # create binning
            return Binning(Binning_Types.USER, edges)
        except ValueError:
            raise click.BadParameter('\'%s\' ' \
                                     'can not be interpreted as list of ' \
                                     'positive floats.' % value)


@click.group()
def cli():
    pass


@cli.group(short_help='subcommand to create things', name='create')
def generate():
    pass


@cli.command(short_help='subcommand to calculate different KPIs',
             name='compare')
@click.argument("real", type=click.Path())
@click.argument("generated", type=click.Path())
@click.argument("binning", callback=validate_binning)
def compare(real, generated, binning):
    # datasources checks
    try:
        real = DataSourceIO.read(real)
    except:
        raise click.FileError(real, "does not exist or is not readable.")
    try:
        generated = DataSourceIO.read(generated)
    except:
        raise click.FileError(generated, "does not exist or is not readable.")

    # validate dimensionality match between real and generated data
    if len(real.domain) != len(generated.domain):
        raise click.UsageError(
            "Dimensions of real data (%d) and generated data (%d) mismatch."
            % (len(real.domain), len(generated.domain)))

    # validate dimensionality match between binning and data
    if binning.dimensions != len(real.domain):
        raise click.UsageError(
            "Dimensions of binning (%d) and data sources (%d) mismatch."
            % (binning.dimensions, len(real.domain)))

    # calculate KPIs
    kpis = KPIs(real, generated, binning)

    click.echo("Error:\t%f" % kpis.error())
    click.echo("Q:\t%f\nQNEB:\t%f\nQEB:\t%f" % kpis.quality())
    click.echo("Point Distance:\t%f" % kpis.distance())


@cli.group(short_help='subcommand to visualize things', name='plot')
def plot():
    pass


@generate.group(short_help='create datasource', name='datasource')
def g_datasource():
    pass


@generate.command(short_help='generate binning', name='binning')
@click.option("--datasource", type=click.Path(), help='path to datasource')
@click.option("--domain", callback=validate_binning_domain,
              help='upper limits for dataset domain (float or list of floats)')
@click.option("--spread", type=float,
              help='spread for irregular binning generation')
@click.argument("type", type=click.Choice(
                [name.lower() for name, value in Binning_Types.__members__.items()
                 if value.value < 90]
                ))
@click.argument("amount", callback=validate_binning_amount)
@click.argument("output", type=click.Path())
def g_binning(datasource, domain, type, amount, output, spread):
    """Generates a binning of a given type, with AMOUNT bins in each dimension.
    The binning is written to OUTPUT in yaml format.

    AMOUNT can be an integer or a comma-separated list of integers, representing
    the number of bins per dimension.

    When not specified, the binning domain is inferred from datasource.
    """
    # datasource = None and domain == None --> Error
    # datasource = None and domain != None --> OK
    # datasource != None and domain = None --> derive domain from datasource
    # datasource != None and domain != None --> dim. should match, use domain
    source = None
    if domain is None and datasource is None:
        raise click.UsageError("Either a datasource or a domain is required.")
    elif not datasource is None:
        try:
            source = DataSourceIO.read(datasource)
        except:
            raise click.FileError(datasource, "does not exist or is not readable.")
        if not domain is None:
            if len(source.domain) != len(domain):
                    raise click.BadOptionUsage("domain",
                                               "Dimensions of datasource domain (%d) and given domain (%d) mismatch."
                                               % (len(source.domain), len(domain)))
        else:
            domain = source.domain

    # convert type to enum
    type = Binning_Types[type.upper()]

    # if type is clustered, datasource is required
    if type == Binning_Types.CLUSTERED and datasource is None:
        raise click.UsageError("Datasource is required for clustered binning.")

    # if spread is not given
    if spread is None:
        binning = BinningGenerator.generate(type, amount, domain, source)
    else:
        binning = BinningGenerator.generate(type, amount, domain, source, spread)

    with open(output, "w") as f:
        yaml.dump(binning.to_dict(), f)
        click.echo("Saved binning to %s" % output)


@generate.command(short_help='derive model', name='model')
@click.option("--padmode", type=click.Choice([
   'epsilon', 'mirror']), default='mirror',
    help='padding mode for 👻 bins, default: mirror')
@click.option("--padvalue", type=click.Choice([
   'zero', 'neg_one', 'copy', 'neg_copy']), default='neg_copy',
   help='padding values for 👻 bins, default: neg_copy')
@click.option("--interpolation", type=click.Choice([
   'linear', 'rbf_linear', 'rbf_multiquad']), default='linear',
   help='model interpolation mode, default: linear')
@click.option("--resource-names", help='comma-separated list of resource names')
@click.argument("datasource", type=click.Path())
@click.argument("binning", callback=validate_binning)
@click.argument("output", type=click.Path())
def g_model(padmode, padvalue, interpolation, resource_names,
            datasource, binning, output):
    """Derives a model from DATASOURCE with given BINNING.
    The model is written to OUTPUT.

    BINNING can be a path to a previously created binning, or custom bin edges
    in all dimension: dimensions are separated by colons, edge values in
    each dimension are separated by commas.
    """
    # datasource checks
    try:
        source = DataSourceIO.read(datasource)
    except:
        raise click.FileError(datasource, "does not exist or is not readable.")

    # validate dimensionality match between binning and source
    if binning.dimensions != len(source.domain):
        raise click.UsageError(
            "Dimensions of binning (%d) and datasource (%d) mismatch."
            % (binning.dimensions, len(source.domain)))

    # resources checks: split list and verify dim match with source
    if not resource_names is None:
        resource_names = resource_names.split(",")
        if len(resource_names) != len(source.column_names):
            raise click.BadOptionUsage("resource-names",
            "Dimensions of resource names (%d) and datasource (%d) mismatch."
            % (len(resource_names), len(source.column_names)))

    # convert model params to enums and create ModelParams object
    model_params = ModelParams(
        Pad_Modes[padmode.upper()],
        Pad_Values[padvalue.upper()],
        Interpolation_Modes[interpolation.upper()]
    )

    # histogram the data with given binning
    histogram = source.get_histogram(binning)

    model = Model.from_histogram(model_params, histogram, resource_names)
    model.to_file(output)


@generate.command(short_help='generate bundles', name='bundles')
@click.option("--use-recommended", is_flag=True, default=False,
    help='use recommended amount of bundles instead of given AMOUNT')
@click.option("--print-ebv", is_flag=True, default=False,
    help='print expected best quality for given AMOUNT')
@click.option("--datasource", type=click.Path(),
    help='path to datasource')
@click.argument("model", type=click.Path(exists=True))
@click.argument("amount", type=int)
@click.argument("binning", callback=validate_binning)
@click.argument("output", type=click.Path())
def g_bundles(use_recommended, print_ebv, datasource,
              model, amount, binning, output):
    """Generates AMOUNT bundles based on MODEL and BINNING.
    The bundles are written to OUTPUT.yaml and OUTPUT.csv.

    MODEL has to be a previously generated model file.

    AMOUNT is an integer that is the amount of bundles that should be
    generated. When --use-recommended is provided, AMOUNT functions as an upper
    limit if set to a value >0.

    BINNING can be a path to a previously created binning, or custom bin edges
    in all dimension: dimensions are separated by colons, edge values in
    each dimension are separated by commas.
    """
    # load model
    try:
        model = Model.from_file(model)
    except Exception:
        # raise this if file exists but does not contan a model
        raise click.FileError(model, 'malformed model file.')

    # check dimensions match for binning and model
    if binning.dimensions != len(model.column_names):
        raise click.UsageError("Dimensions of binning (%d) and model (%d) mismatch."
                               % (binning.dimensions, len(model.domain)))

    # if ebv and recommended amount are requested, real datasource is required
    # load datasource and histogram it using the desired binning
    if use_recommended or print_ebv:
        if datasource is None:
            raise click.UsageError("Datasource required for --use-recommended and --print-ebv options.")
        try:
            source = DataSourceIO.read(datasource)
        except:
            raise click.FileError(datasource, "does not exist or is not readable.")
        real_histogram = source.get_histogram(binning)

    # option to generate uniform prob instead of using model

    # create DatasetGenerator
    bg = BundleGenerator(model, binning)

    # create dict for additional information
    addi = {}

    # use recommended amount
    if use_recommended:
        recommended = bg.recommended_amount(real_histogram)
        if amount <= 0:
            amount = recommended
        else:
            amount = min(amount, recommended)
        addi["recommended"] = recommended
        click.echo("Using recommended amount: %d" % amount)

    ebv = bg.expected_best_quality(amount, real_histogram)
    addi["ebv"] = np.float(ebv)
    addi["amount"] = amount

    # print ebv
    if print_ebv:
        click.echo("Expected best quality: %f" % ebv)

    # generate bundles and save to OUTPUT
    bundles = bg.generate(amount)
    DataSourceIO.write(bundles,
                       output,
                       additional_info={"metadata_from_bundle_generation": addi})


class G_DATASOURCE():

    @staticmethod
    @g_datasource.command(short_help='generate a dataset derived from the Google cloud traces')
    @click.option("--name", default="", help="Logical name of the dataset.")
    @click.argument("output", type=click.Path())
    @click.argument("input", type=click.Path(), nargs=-1, required=True)
    def google(name, output, input):
        """Generates a dataset that is derived from the Google Cluster Data workload
           traces[1] (ClusterData2011_2) found in the INPUT files
           (note: multiple input files can be given).

           The data is written to "OUTPUT.csv", the metadata to "OUTPUT.yaml".

           [1] https://github.com/google/cluster-data
        """
        GoogleDatasetProcessor(name=name,
                               output_filename=output,
                               source_filenames=input).process()

    @staticmethod
    @g_datasource.command(short_help='generate a dataset derived from the Bitbrains log format')
    @click.option("--name", default="", help="Logical name of the dataset.")
    @click.argument("output", type=click.Path())
    @click.argument("input", type=click.Path())
    def bitbrains(name, output, input):
        """Generates a dataset that is derived from the BitBrains fastStorage trace
           data[1] found in folder INPUT.

           The data is written to "OUTPUT.csv", the metadata to "OUTPUT.yaml".

           [1] http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains
        """
        BitbrainsDatasetProcessor(name=name,
                                  output_filename=output,
                                  source_folder=input).process()

    @staticmethod
    @g_datasource.command(short_help='generate a uniform dataset')
    @click.option("--name", default="", help="Logical name of the dataset.")
    @click.argument("output", type=click.Path())
    @click.argument("dimensions", type=int)
    def uniform(name, output, dimensions):
        """Generates a DIMENSIONS-dimensional dataset consisting of uniformly
           distributed datapoints.

           The data is written to "OUTPUT.csv", the metadata to "OUTPUT.yaml".
        """
        UniformDatasetProcessor(name=name,
                                output_filename=output,
                                dimensions=dimensions).process()

    @staticmethod
    @g_datasource.command(short_help='generate a dataset based on hotspots')
    @click.option("--name", default="", help="Logical name of the dataset.")
    @click.argument("output", type=click.Path())
    @click.argument("dimensions", type=int)
    @click.argument("hotspots", type=int)
    def hotspots(name, output, dimensions, hotspots):
        """Generates a DIMENSIONS-dimensional dataset consisting
           of HOTSPOTS amount of hotspots.

           The data is written to "OUTPUT.csv", the metadata to "OUTPUT.yaml".
        """
        HotspotsDatasetProcessor(name=name,
                                 output_filename=output,
                                 dimensions=dimensions,
                                 hotspot_count=hotspots).process()


@plot.command(short_help='plot datasource', name="data")
@click.option("--show", is_flag=True, default=False, help='show figure')
@click.option("--cmap", default='Blues', help='matplotlib colormap name')
@click.option("--title", help='title to be displayed above figure')
@click.option("--resource-names", help='comma-separated list of resource names')
@click.argument("datasource", type=click.Path())
@click.argument("binning", callback=validate_binning)
@click.argument("output", type=click.Path())
def p_data(show, cmap, title, resource_names, datasource, binning, output):
    """Plots a histogram of DATASOURCE, binned using BINNING.
    The figure is saved to OUTPUT.png.

    For multi-dimensional data, each pair of resources is plotted as a 2D histogram.

    BINNING can be a path to a previously created binning, or custom bin edges
    in all dimension: dimensions are separated by colons, edge values in
    each dimension are separated by commas.
    """
    try:
        cmap = plt.get_cmap(cmap)
    except Exception:
        raise click.BadOptionUsage("cmap",
            "must be a valid Matplotlib colormap name, not \'%s\'."
            % cmap)

    try:
        source = DataSourceIO.read(datasource)
    except:
        raise click.FileError(datasource, "does not exist or is not readable.")

    # validate dimensionality match between binning and source
    if binning.dimensions != len(source.domain):
        raise click.UsageError(
            "Dimensions of binning (%d) and datasource (%d) mismatch."
            % (binning.dimensions, len(source.domain)))

    # resources checks: split list and verify dim match with source
    if resource_names is None:
        resource_names = source.column_names
    else:
        resource_names = resource_names.split(",")
        if len(resource_names) != len(source.column_names):
            raise click.BadOptionUsage("resource-names",
            "Dimensions of resource names (%d) and datasource (%d) mismatch."
            % (len(resource_names), len(source.column_names)))

    histogram = source.get_histogram(binning)

    HairyPlotter.plot_histogram(
        histogram,
        cmap=cmap,
        column_names=resource_names,
        title=title)

    plt.savefig(output, bbox_inches='tight')

    if show:
        plt.show()


@plot.command(short_help='plot model', name="model")
@click.option("--show", is_flag=True, default=False, help='show figure')
@click.option("--cmap", default='Blues', help='matplotlib colormap name')
@click.option("--title", help='title to be displayed above figure')
@click.option("--resource-names", help='comma-separated list of resource names')
@click.argument("model", type=click.Path(exists=True))
@click.argument("binning", callback=validate_binning)
@click.argument("output", type=click.Path())
def p_model(show, cmap, title, resource_names, model, binning, output):
    """Plots probabilities derived by MODEL, histogrammed using BINNING.
    The figure is saved to OUTPUT.png.

    For multi-dimensional data, each pair of resources is plotted as a 2D histogram.

    BINNING can be a path to a previously created binning, or custom bin edges
    in all dimension: dimensions are separated by colons, edge values in
    each dimension are separated by commas.
    """
    try:
        cmap = plt.get_cmap(cmap)
    except Exception:
        raise click.BadOptionUsage("cmap",
            "must be a valid Matplotlib colormap name, not \'%s\'."
            % cmap)

    # load model
    try:
        model = Model.from_file(model)
    except Exception:
        # raise this if file exists but does not contan a model
        raise click.FileError(model, 'malformed model file.')

    # check dimensions match for binning and model
    if binning.dimensions != len(model.column_names):
        raise click.UsageError("Dimensions of binning (%d) and model (%d) mismatch."
                               % (binning.dimensions, len(model.domain)))

    # resources checks: split list and verify dim match with source
    if resource_names is None:
        resource_names = model.column_names
    else:
        resource_names = resource_names.split(",")
        if len(resource_names) != len(model.column_names):
            raise click.BadOptionUsage("resource-names",
            "Dimensions of resource names (%d) and datasource (%d) mismatch."
            % (len(resource_names), len(model.column_names)))

    HairyPlotter.plot_model(
        model, binning,
        cmap=cmap,
        column_names=resource_names,
        title=title)

    plt.savefig(output, bbox_inches='tight')

    if show:
        plt.show()


if __name__ == '__main__':
    cli()
