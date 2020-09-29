import datasets.Dataset
import datasets.handlers.extractImages
import datasets.handlers.extractLabels
import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.initializers.HeNormal
import api.keras.initializers.Zeros
import api.keras.layers.Dense
import api.keras.layers.Input
import api.keras.layers.Flatten
import api.keras.optimizers.SGD
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics

private val SEED = 42L

private val model = Sequential.of(
        Input(28,28),
        Flatten(),
        Dense(300, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
        Dense(100, Activations.Relu, kernelInitializer  = HeNormal(SEED), biasInitializer = Zeros()),
        Dense(10, Activations.HardSigmoid, kernelInitializer  = HeNormal(SEED), biasInitializer = Zeros())
)

val (train, test) = Dataset.createTrainAndTestDatasets(
            "MNIST/train-images-idx3-ubyte.gz",
            "MNIST/train-labels-idx1-ubyte.gz",
            "MNIST/t10k-images-idx3-ubyte.gz",
            "MNIST/t10k-labels-idx1-ubyte.gz",
            10,
            ::extractImages,
            ::extractLabels
    )
    val (newTrain, validation) = train.split(0.95)

    model.compile(SGD(), LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)
    model.summary()

    model.fit(dataset = train, epochs = 30, batchSize = 100, verbose = true)

    val accuracy = model.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

    model.close()

println("Accuracy: $accuracy")

