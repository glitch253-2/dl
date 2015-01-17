import breeze.linalg.{DenseMatrix, DenseVector}
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
    val m = 60000
    val n = (28 * 28)
    val features = DenseMatrix.zeros[Double](m,n)
    println(features.rows)
    println(features.cols)
    val classes = DenseVector.zeros[Double](60000)
    (0 until m).foreach(idx => {
      val row = new DenseVector[Double](trainingData(idx)._1)
      features(idx, ::) := row.t
      classes(idx) = trainingData(idx)._2
    })
    val models = (0 to 9).map(idx => {
      val model = new Logistic(10, 150)
      val labels = DenseVector.zeros[Double](60000)
      (0 until m).foreach(label => {
        if (classes(label) == idx)
          labels(label) = 1.0
        else labels(label) = 0.0
      })
      model.train(features.copy, labels)
      model
    })

    val pw = new PrintWriter(new File("output.txt"))
    var numCorrect = 0
    val test = testingData.map(observation => {
      val classProbabilities = (0 to 9).map(idx => {
        models(idx).predict(new DenseVector(observation._1), Some(observation._2))
      })
      classProbabilities
    }).map(prediction => {
      val results = prediction.map(_._1)
      val index = results.zipWithIndex.maxBy(_._1)._2
      if (index.toDouble == prediction(0)._2) {
        numCorrect = numCorrect + 1
      }
      s"Prediction: ${index.toDouble} Actual: ${prediction(0)._2} "
    }).foreach(pw.println(_))
    pw.println("Number correct: " + numCorrect)
    pw.println("Positive rate: " + numCorrect / 10000.0)
    pw.close()
  }
}
