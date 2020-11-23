import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Flatten
import org.jetbrains.kotlinx.dl.api.core.layer.Input
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsByPathTemplates
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayersByPathTemplates
import org.jetbrains.kotlinx.dl.api.inference.keras.recursivePrintGroupInHDF5File
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.awt.image.DataBufferByte
import java.io.File


val SEED = 42L

private fun prepareCustomDatasetFromPaths(vararg paths: String): Dataset {
    val listOfImages = mutableListOf<FloatArray>()
    val listOfLabels = mutableListOf<FloatArray>()
    val numberOfClasses = paths.size
    var counter = 0

    for (path in paths) {
        File(path).walk().forEach {
            try {
                val image = ImageConverter.toNormalizedFloatArray(it)
                listOfImages.add(image)
                val label = FloatArray(numberOfClasses)
                label[counter] = 1F
                listOfLabels.add(label)
            } catch (e: Exception) {
                println("Skipping the following image $it")
            }
        }
        counter += 1
    }

    val sortedData = listOfImages.zip(listOfLabels)
    val shuffledData = sortedData.shuffled()
    val (x, y) = shuffledData.unzip()

    return Dataset.create({ x.toTypedArray() }, { y.toTypedArray() })
}

fun main() {

    val data = prepareCustomDatasetFromPaths(
        "src/main/resources/eagles-pigeons/eagles224x224",
        "src/main/resources/eagles-pigeons/pigeons224x224"
    )

    val (train, test) = data.split(0.8)

    val WEIGHTS = File("src/main/resources/VGG19/weights.h5")
    val VGG19_CONFIG = File("src/main/resources/VGG19/model.json")
    val hdfFile = HdfFile(WEIGHTS)

    val (input, otherLayers) = Sequential.loadModelLayersFromConfiguration(VGG19_CONFIG)

    val layers = mutableListOf<Layer>()

    for (layer in otherLayers) {
        if (layer::class == Conv2D::class || layer::class == MaxPool2D::class) {
            layer.isTrainable = false
            layers.add(layer)
        }
        else {layer.isTrainable = true
            layers.add(layer)
        }
    }

    layers.removeLast()

    layers.add(
        Dense(
            name = "new_prediction_layer",
            kernelInitializer = GlorotUniform(SEED),
            biasInitializer = GlorotUniform(SEED),
            outputSize = 2,
            activation = Activations.Linear
        )
    )

    val model = Sequential.of(input, layers)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.summary()

        it.loadWeightsForFrozenLayers(hdfFile)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            batchSize = 32,
            epochs = 3,
            verbose = false
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}