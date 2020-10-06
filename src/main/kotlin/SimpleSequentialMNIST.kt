import api.core.SavingFormat
import api.core.Sequential
import api.core.activation.Activations
import api.core.initializer.HeNormal
import api.core.initializer.Zeros
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Input
import api.core.loss.LossFunctions
import api.core.metric.Metrics
import api.core.optimizer.SGD
import datasets.Dataset
import datasets.handlers.extractImages
import datasets.handlers.extractLabels
import java.io.File

private val SEED = 42L

private val model = Sequential.of(
    Input(28,28),
    Flatten(),
    Dense(300, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(100, Activations.Relu, kernelInitializer  = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(10, Activations.HardSigmoid, kernelInitializer  = HeNormal(SEED), biasInitializer = Zeros())
)

fun main() {
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

    model.fit(dataset = newTrain, epochs = 30, batchSize = 100, verbose = true)

    val accuracy = model.evaluate(dataset = validation, batchSize = 100).metrics[Metrics.ACCURACY]

    model.save(File("src/model/my_model"))
    model.close()


    println("Accuracy: $accuracy")

}