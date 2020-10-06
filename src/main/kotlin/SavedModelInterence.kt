import api.inference.InferenceModel
import org.tensorflow.Tensor
import sun.awt.resources.awt
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import javax.swing.Spring.height


val PATH_TO_MODEL = "src/model/my_saved_model"
val image = ImageIO.read(File("/Users/mariakhalusova/Code/KDL-Playground/src/main/resources/testing-resources/test-image.png"))

val imageData = image.getData()

private fun reshape(floats: FloatArray): Tensor<*> {
    val reshaped = Array(
        1
    ) { Array(28) { FloatArray(28) } }
    for (i in floats.indices) reshaped[0][i / 28][i % 28] = floats[i]
    return Tensor.create(reshaped)
}


fun main() {

    val pixels = IntArray(imageData.width * imageData.height)
    val ar = imageData.let {
        it.getPixels(0, 0, it.width, it.height, pixels)
    }
    val flarr: FloatArray = ar.map{it.toFloat()}.toFloatArray()
    //I'd love to be able to have Keras style image pre-processing functionality, e.g. image as array https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img

    InferenceModel().use {
        it.reshape(::reshape)
        it.loadWeights(File("src/model/my_model"))
        println(it.predict(flarr))
    }

}

