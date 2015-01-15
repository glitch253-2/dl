import breeze.linalg._
import breeze.numerics._

case class Logistic(lr: Double = 0.95, iterations: Int = 100) extends Classifier {
  var weights = new DenseVector[Double](Array(1.0))
  var learningRate = lr
  var means = new DenseVector[Double](Array(1.0))

  private def cost(theta: DenseVector[Double], x: DenseMatrix[Double], classes: DenseVector[Double]): Double = {
      val hTheta = sigmoid(((theta * -1.0).t * x.t).inner)
      sum(classes :* log(hTheta) + ((classes * -1.0)  + 1.0) :* log((hTheta * -1.0) + 1.0)) * (1.0/x.rows)
  }

  private def computeUpdates(theta: DenseVector[Double], x: DenseMatrix[Double], classes: DenseVector[Double]) = {
    val update = DenseVector.zeros[Double](theta.length)
    val hTheta = sigmoid(((theta * -1.0).t * x.t).inner)
    (0 until theta.length).foreach(col => {
      var singleUpdate = learningRate * (-1.0 / x.rows)
      var multiplier = 0.0
      (0 until x.rows).foreach(row => {
        multiplier = multiplier + ((hTheta(col) - classes(col)) * x(row, col))
      })
      update(col) = update(col) + singleUpdate * multiplier
    })
    update
  }

  def train(features: DenseMatrix[Double], labels: DenseVector[Double]) = {

    val means = DenseVector.zeros[Double](features.cols)

    val stds = DenseVector.zeros[Double](features.cols)
    (0 until features.cols).foreach(idx => {
      means(idx) = sum(features(idx,::).inner) / features.rows
      stds(idx) = Math.sqrt(sum(features(idx,::).inner + -means(idx) :* (features(idx,::).inner + -means(idx))))
    })

    (0 until features.cols).foreach(idx => {
      features(::,idx) := (features(::,idx) - means(idx)) / stds(idx)
    })

    weights = DenseVector.zeros[Double](features.cols)
    (0 until iterations).foreach(iteration => {
      println(cost(weights, features, labels))
      weights = computeUpdates(weights, features, labels)
      learningRate = learningRate * ((iterations - Math.log(iteration + 1)) / iterations.toDouble)
     })
  }

  def predict(observation: DenseVector[Double], trueLabel: Option[Double]): (Double, Double) = {
    (sigmoid(((weights * -1.0).t * observation)), trueLabel.getOrElse(Double.MaxValue))
  }

}