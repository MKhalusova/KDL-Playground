import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Flatten
import org.jetbrains.kotlinx.dl.api.core.layer.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.handlers.extractImages
import org.jetbrains.kotlinx.dl.datasets.handlers.extractLabels
import java.io.File

private val model = Sequential.of(
    Input(28,28,1),
    Flatten(),
    Dense(300),
    Dense(100),
    Dense(10)
)

fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        trainFeaturesPath = "datasets/mnist/train-images-idx3-ubyte.gz",
        trainLabelsPath = "datasets/mnist/train-labels-idx1-ubyte.gz",
        testFeaturesPath = "datasets/mnist/t10k-images-idx3-ubyte.gz",
        testLabelsPath = "datasets/mnist/t10k-labels-idx1-ubyte.gz",
        numClasses = 10,
        ::extractImages,
        ::extractLabels
    )
    val (newTrain, validation) = train.split(splitRatio = 0.95)

    model.use{
        it.compile(optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY)

        it.summary()

        it.fit(dataset = newTrain,
                epochs = 10,
                batchSize = 100,
                verbose = false)

        val accuracy = it.evaluate(dataset = validation,
                batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
        it.save(File("src/model/my_model"))
    }
}