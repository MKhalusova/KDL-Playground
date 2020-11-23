import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.io.File

val PATH_TO_MODEL = "src/model/my_model"
val PATH_TO_TEST_IMAGE = "src/main/resources/testing-resources/test-image-bag.png"
val stringLabels = mapOf(0 to "T-shirt/top",
        1 to "Trouser",
        2 to "Pullover",
        3 to "Dress",
        4 to "Coat",
        5 to "Sandal",
        6 to "Shirt",
        7 to "Sneaker",
        8 to "Bag",
        9 to "Ankle boot"
)

val floatArray = ImageConverter.toRawFloatArray(File(PATH_TO_TEST_IMAGE))

fun reshapeInput(inputData: FloatArray): Array<Array<FloatArray>> {
    val reshaped = Array(
            1
    ) { Array(28) { FloatArray(28) } }
    for (i in inputData.indices) reshaped[0][i / 28][i % 28] = inputData[i]
    return reshaped
}

fun main() {
    InferenceModel.load(File(PATH_TO_MODEL)).use {
        it.reshape(::reshapeInput)
        val prediction = it.predict(floatArray)
        println("Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}.")
    }

}

