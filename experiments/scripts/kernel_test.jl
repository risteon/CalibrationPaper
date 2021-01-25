#' ---
#' title: Calibration error estimates and p-value approximations for modern neural networks
#' author: David Widmann
#' ---

#' # Intro
#'
#' In the following experiments we download a set of pretrained modern neural networks for
#' the image data set CIFAR-10. We estimate the expected calibration error (ECE) with
#' respect to the total variation distance and the squared kernel calibration error of
#' these models.

#' # Packages
#'
#' We perform distributed computing to speed up our computations.

using Distributed

#' First we have to activate the local package environment on all cores.

@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

#' Then we load the packages that are required on all cores.

@everywhere begin
    using CalibrationErrors
    using CalibrationPaper
    using CalibrationTests

    using DelimitedFiles
    using Random
end

#' The following packages are only required on the main process.

using Conda
using CSV
using DataDeps
using DataFrames
using ProgressMeter
using PyCall

using LibGit2

#' # Experiments
#'
#' ## Pretrained neural networks
#'
#' As a first step we download a set of pretrained neural networks for CIFAR-10 from
#' [PyTorch-CIFAR10](https://github.com/huyvnphan/PyTorch-CIFAR10). We extract the
#' predictions of these models on the validation data set together with the correct labels.
#' First we check if the predictions and labels are already extracted.

# create directory for results
const DATADIR = joinpath(@__DIR__, "..", "data", "scssnet-KITTI_01")
# const DATADIR = joinpath(@__DIR__, "..", "data", "scssnet-KITTI_25")
isdir(DATADIR) || mkpath(DATADIR)

# dl = 55962140
dl = 2313196

# check if predictions exist
# const ALL_MODELS = ["upgrade_v1_c000054", "uncertainty_focal_v1_c000067", "uncertainty_heteroscedastic_c000146", "uncertainty_visibility_c000073"]
const ALL_MODELS = ["upgrade_v1_c000054"]
const MISSING_MODELS = filter(ALL_MODELS) do name
    !isfile(joinpath(DATADIR, "$name.bin"))
end

#' If the data does not exist, we start by loading all missing packages and
#' registering the required data. If you want to rerun this experiment from
#' scratch, please download the pretrained weights of the models.

if !isfile(joinpath(DATADIR, "labels.bin"))   
    throw(ArgumentError("Missing labels file."))
end

if !isempty(MISSING_MODELS)
    throw(ArgumentError("Missing predictions file."))
end

labels = Array{UInt8}(undef, dl);
read!(joinpath(DATADIR, "labels.bin"), labels) # read data
#' Debug prints
println(size(labels))
println(length(labels))

const LABELS = labels
# println("Max")
# println(maximum(LABELS))

datadir = DATADIR
labels = LABELS

for model in ALL_MODELS
    # load predictions
    rawdata = Array{Float16}(undef, dl, 20);
    read!(joinpath(datadir, "$model.bin"), rawdata) # read data


    rawdata_t = reshape(rawdata, dl * 20)
    rawdata_t = resize!(rawdata_t, 10000 * 20)
    rawdata_t = reshape(rawdata_t, :, 20)

    predictions_t = [convert(Array{Float64}, rawdata_t[i, :]) for i in axes(rawdata_t, 1)]

    println("pos A")
    predictions = [convert(Array{Float64}, rawdata[i, :]) for i in axes(rawdata, 1)]
    println("pos B")

    # compute kernel based on the median heuristic
    # kernel = median_TV_kernel(predictions_t)
    γ = debug_bla(predictions_t)
    
    println("Gamma:", γ)
end
