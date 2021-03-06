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
    using Distances

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
# const DATADIR = joinpath(@__DIR__, "..", "data", "scssnet-KITTI_01")
const DATADIR = joinpath(@__DIR__, "..", "data", "scssnet-KITTI_25")

isdir(DATADIR) || mkpath(DATADIR)

dl = 55962140
# dl = 2313196

# check if predictions exist
const ALL_MODELS = ["upgrade_v1_c000054", "uncertainty_focal_v1_c000067", "uncertainty_heteroscedastic_c000146", "uncertainty_visibility_c000073"]
# const ALL_MODELS = ["upgrade_v1_c000054"]

const MISSING_MODELS = filter(ALL_MODELS) do name
    !isfile(joinpath(DATADIR, "$name.bin"))
end

#' If the data does not exist, we start by loading all missing packages and
#' registering the required data. If you want to rerun this experiment from
#' scratch, please download the pretrained weights of the models.

if !isempty(MISSING_MODELS) || !isfile(joinpath(DATADIR, "labels.bin"))   
    throw(ArgumentError("Missing labels file."))
end

# throw(ArgumentError("Ending here."))

#' We load the true labels since they are the same for every model.

labels = Array{UInt8}(undef, dl);
read!(joinpath(DATADIR, "labels.bin"), labels) # read data
#' Debug prints
println(size(labels))
println(length(labels))

const LABELS = labels .+ 1
println("Largest label: ", maximum(LABELS))
println("Smallest label: ",minimum(LABELS))

# const LABELS = CSV.read(joinpath(DATADIR, "labels.csv");
#     header = false, delim = ',', type = Int) |> Matrix{Int} |> vec

# ' ## Calibration tests
# '
# ' Additionally we compute different p-value approximations for each model. More concretely,
# ' we estimate the p-value by consistency resampling of the two ECE estimators mentioned
# ' above, by distribution-free bounds of the three SKCE estimators discussed above, and by
# ' the asymptotic approximations for the unbiased quadratic and linear SKCE estimators
# ' used above. The results are saved in a CSV file `pvalues.csv`.

@everywhere function calibration_pvalues(rng::AbstractRNG, predictions, labels, kernel, channel)
    # evaluate consistency resampling based estimators
    # ece_uniform = ConsistencyTest(ECE(UniformBinning(10)), predictions, labels)
    # pvalue_ece_uniform = pvalue(ece_uniform; rng = rng)
    put!(channel, true)
    # ece_dynamic = ConsistencyTest(ECE(MedianVarianceBinning(100)), predictions, labels)
    # pvalue_ece_dynamic = pvalue(ece_dynamic; rng = rng)
    put!(channel, true)

    # compute kernel based on the median heuristic
    # kernel = median_TV_kernel(predictions)

    # evaluate distribution-free bounds
    # skceb_median_distribution_free = DistributionFreeTest(BiasedSKCE(kernel), predictions, labels)
    # pvalue_skceb_median_distribution_free = pvalue(skceb_median_distribution_free)
    put!(channel, true)
    # skceuq_median_distribution_free = DistributionFreeTest(QuadraticUnbiasedSKCE(kernel), predictions, labels)
    # pvalue_skceuq_median_distribution_free = pvalue(skceuq_median_distribution_free)
    put!(channel, true)
    skceul_median_distribution_free = DistributionFreeTest(LinearUnbiasedSKCE(kernel), predictions, labels)
    pvalue_skceul_median_distribution_free = pvalue(skceul_median_distribution_free)
    put!(channel, true)

    # evaluate asymptotic bounds
    # skceuq_median_asymptotic = AsymptoticQuadraticTest(kernel, predictions, labels)
    # pvalue_skceuq_median_asymptotic = pvalue(skceuq_median_asymptotic; rng = rng)
    put!(channel, true)
    skceul_median_asymptotic = AsymptoticLinearTest(kernel, predictions, labels)
    pvalue_skceul_median_asymptotic = pvalue(skceul_median_asymptotic)
    put!(channel, true)

    (
        # ECE_uniform = pvalue_ece_uniform,
        # ECE_dynamic = pvalue_ece_dynamic,
        # SKCEb_median_distribution_free = pvalue_skceb_median_distribution_free,
        # SKCEuq_median_distribution_free = pvalue_skceuq_median_distribution_free,
        SKCEul_median_distribution_free = pvalue_skceul_median_distribution_free,
        # SKCEuq_median_asymptotic = pvalue_skceuq_median_asymptotic,
        SKCEul_median_asymptotic = pvalue_skceul_median_asymptotic,
    )
end

# do not recompute the p-values if a file with results exists
if isfile(joinpath(DATADIR, "pvalues.csv"))
    @info "skipping p-value approximations: output file $(joinpath(DATADIR, "pvalues.csv")) exists"
else
    # define the pool of workers, the progress bar, and its update channel
    wp = CachingPool(workers())
    n = length(ALL_MODELS)
    p = Progress(7 * n, 1, "computing p-value approximations...")
    channel = RemoteChannel(() -> Channel{Bool}(7 * n))

    local estimates
    @sync begin
        # update the progress bar
        @async while take!(channel)
            next!(p)
        end

        # compute the p-value approximations for all models
        estimates = let rng = Random.GLOBAL_RNG, datadir = DATADIR, labels = LABELS, channel = channel
            pmap(wp, ALL_MODELS) do model
                # # load predictions
                # rawdata = CSV.read(joinpath(datadir, "$model.csv");
                #                    header = false, transpose = true, delim = ',',
                #                    type = Float64) |> Matrix{Float64}
                # predictions = [rawdata[:, i] for i in axes(rawdata, 2)]

                # load predictions
                rawdata = Array{Float16}(undef, 20, dl);
                read!(joinpath(datadir, "$model.bin"), rawdata) # read data

                median_value = readdlm(joinpath(datadir, "median_$model.ascii"), '\t', Float64, '\n')
                median_value = median_value[1]
                println("Loaded median value: ", median_value)

                # predictions = [convert(Array{Float64}, rawdata[:, i]) for i in axes(rawdata, 2)]
                predictions = [rawdata[:, i] for i in axes(rawdata, 2)]

                # copy random number generator and set seed
                _rng = deepcopy(rng)
                Random.seed!(_rng, 1234)

                # compute kernel based on the median heuristic
                # kernel = median_TV_kernel(predictions_t)
                # pre-computed median value
                γ = inv(median_value)
                kernel = UniformScalingKernel(ExponentialKernel(γ, TotalVariation()))

                # compute approximations
                pvalues = calibration_pvalues(_rng, predictions, labels, kernel, channel)
                merge((model = model,), pvalues)
            end
        end

        # stop progress bar
        put!(channel, false)
    end

    # save estimates
    @info "saving p-value approximations..."
    CSV.write(joinpath(DATADIR, "pvalues.csv"), estimates)
end
