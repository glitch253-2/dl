import breeze.linalg.DenseVector

/**
 * Created by Mike on 1/14/2015.
 */
object nist {
  def main(args: Array[String]) = {
    val inputs = scala.io.Source.fromFile(args(0)).getLines.map(_.split(args(1)).map(_.toDouble)).toList
    val classIndex = args(2).toInt
    val classes = inputs.map(_(classIndex))
    val features: List[(DenseVector[Double], Double)] = inputs.map(row => {
      new DenseVector[Double](row.dropRight(10))
    }).zip(classes)
  }
}
