import breeze.linalg.DenseVector
import org.deeplearning4j.datasets.mnist._
import java.io._
/**
 * Created by mike on 1/14/2015.
 */
object MNIST {
  val mnistTraining = new MnistManager("train-images.idx3-ubyte",
    "train-labels.idx1-ubyte")

  val mnistTesting = new MnistManager("t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte")

  def getDataSet(mnist: MnistManager) = {
    val ImageManager = mnist.getImages
    val LabelManager = mnist.getLabels
    val labelledData = (1 until ImageManager.getCount() + 1)
      .map(idx => {
      ImageManager.setCurrentIndex(idx.toLong)
      LabelManager.setCurrentIndex(idx.toLong)
      (ImageManager.readImage().flatMap(_.map(_.toDouble)), LabelManager.readLabel().toDouble)
    })
    labelledData
  }



  def main(args: Array[String]) = {
    val trainingData = getDataSet(mnistTraining)
    val testingData = getDataSet(mnistTesting)
    val models = (0 until 9).map(idx => {
      val model = new Logistic(0.01, 15)
      model.train(trainingData.map(observation => {
        if (observation._2 == idx)
          (new DenseVector(observation._1), 1.0)
        else (new DenseVector(observation._1), 0.0)
      }))
      model
    })

    val pw = new PrintWriter(new File("output.txt"))
    val test = testingData.map(observation => {
      val classProbabilities = (0 until 9).map(idx => {
        models(idx).predict(new DenseVector(observation._1), Some(observation._2))
      })
      classProbabilities
    }).map(prediction => {
      s"Actual: ${prediction(0)._2} " +
      s"0: ${prediction(0)._1} " +
      s"1: ${prediction(1)._1} " +
      s"2: ${prediction(2)._1} " +
      s"3: ${prediction(3)._1} " +
      s"4: ${prediction(4)._1} " +
      s"5: ${prediction(5)._1} " +
      s"6: ${prediction(6)._1} " +
      s"7: ${prediction(7)._1} " +
      s"8: ${prediction(8)._1} " +
      s"9: ${prediction(9)}._1 "
    }).foreach(pw.println(_))
    pw.close()
  }
}
