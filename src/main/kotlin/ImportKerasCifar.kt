import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.io.File

val labelsMap = mapOf(
    0 to "airplane",
    1 to "automobile",
    2 to "bird",
    3 to "cat",
    4 to "deer",
    5 to "dog",
    6 to "frog",
    7 to "horse",
    8 to "ship",
    9 to "truck"
)

val PATH_TO_MODEL_JSON = "src/main/resources/KERAS-CIFAR-10/model.json"
val PATH_TO_WEIGHTS = "src/main/resources/KERAS-CIFAR-10/weights.h5"
val PATH_TO_IMAGE = "src/main/resources/testing-resources/cifar-test-images/test-airplane.png"

val imageArray = ImageConverter.toNormalizedFloatArray(File(PATH_TO_IMAGE))

fun main() {
    val JSONConfig = File(PATH_TO_MODEL_JSON)
    val weights = File(PATH_TO_WEIGHTS)

    val model = Sequential.loadModelConfiguration(JSONConfig)

    model.use {
        it.compile(Adam(), Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)
        val hdfFile = HdfFile(weights)
        it.loadWeights(hdfFile)

        val prediction = it.predict(imageArray)
        println("Predicted label is: $prediction. This corresponds to class ${labelsMap[prediction]}.")
    }
}