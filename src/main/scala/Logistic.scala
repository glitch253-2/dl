import breeze.linalg._
import breeze.numerics._

case class Logistic(lr: Double = 0.95, iterations: Int = 100, coolDown: Double = .9) extends Classifier {
  var weights = new DenseVector[Double](Array(1.0))
  var learningRate = lr
  var means = new DenseVector[Double](Array(1.0))
  var stds = new DenseVector[Double](Array(1.0))

  private def cost(theta: DenseVector[Double], x: DenseMatrix[Double], classes: DenseVector[Double]): Double = {
      val hTheta = hThetaVec(theta, x)
      sum(pow((hTheta - classes), 2)) * (1.0 / 2.0 * x.rows)
  }

  private def hThetaVec(theta: DenseVector[Double], x: DenseMatrix[Double]): DenseVector[Double] = {
    val result = DenseVector.zeros[Double](x.rows)
    (0 until x.rows).foreach(idx => {
      val instance = x(idx, ::).inner
      val intermediate = (theta.t * instance)
      result(idx) = 1.0 / (1.0 + Math.exp(-intermediate))
    })
    result
  }

  private def computeUpdates(theta: DenseVector[Double], x: DenseMatrix[Double], classes: DenseVector[Double]) = {
    val update = theta.copy
    val hTheta = hThetaVec(theta, x)
    (0 until theta.length).foreach(col => {
      var singleUpdate = learningRate
      var multiplier = 0.0
      (0 until x.rows).foreach(row => {
        multiplier = multiplier + ((hTheta(row) - classes(row)) * x(row, col))
      })
      update(col) = update(col) - singleUpdate * multiplier
    })
    update
  }

  def train(features: DenseMatrix[Double], labels: DenseVector[Double]) = {
    /*means = DenseVector.zeros[Double](features.cols)
    stds = DenseVector.zeros[Double](features.cols)
    (0 until features.cols).foreach(idx => {
      means(idx) = sum(features(idx,::).inner) / features.rows
      stds(idx) = Math.sqrt(sum(features(idx,::).inner + -means(idx) :* (features(idx,::).inner + -means(idx))))
    })

    (0 until features.cols).foreach(idx => {
      features(::,idx) := (features(::,idx) - means(idx)) / stds(idx)
    })*/

    weights = DenseVector.rand[Double](features.cols)
    (0 until iterations).foreach(iteration => {
      println("Current Cost: " + cost(weights, features, labels) + " Current Learning Rate: " + learningRate)
      weights = computeUpdates(weights, features, labels)
      if (iteration / iterations > .9) // if we're about 90% of the way through the descent, then we start cooling off
        // the learning rate
        learningRate = learningRate * ((iterations - Math.log(iteration + 1)) / iterations.toDouble)

     })
  }

  def predict(observation: DenseVector[Double], trueLabel: Option[Double]): (Double, Double) = {
    //val normalizedObservation = (observation - means) :/ stds
    (sigmoid(((weights).t * observation)), trueLabel.getOrElse(Double.MaxValue))
  }

}