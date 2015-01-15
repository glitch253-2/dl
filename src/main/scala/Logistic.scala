import breeze.linalg.DenseVector

case class Logistic(lr: Double = 0.95, iterations: Int = 100) extends Classifier {
  var weights = new DenseVector[Double](Array(1.0))
  var learningRate = lr

  private def sigmoid(theta: DenseVector[Double], x: DenseVector[Double]) = {
    val innerProduct = theta.t * x
    1.0 / ( 1.0 + Math.exp(-innerProduct))
  }

  private def costSingle(theta: DenseVector[Double], x:DenseVector[Double], truth: Double): Double = {
    (-truth * Math.log(sigmoid(theta, x)) - (1 - truth) * Math.log(1 - sigmoid(theta, x)))
  }

  private def cost(weights: DenseVector[Double], observations: Seq[(DenseVector[Double], Double)]): Double = {
    (1.0 / observations.size) * observations.map(observation => {
      costSingle(observation._1, weights, observation._2)
    }).reduce(_ + _)
  }

  private def computeUpdates(weights: DenseVector[Double], observations: Seq[(DenseVector[Double], Double)]) = {
    (0 until weights.length).foreach(idx => {
      weights(idx) = weights(idx) - learningRate * observations.map(observation => (sigmoid(weights, observation._1) -
        observation._2) * observation._1.data(idx)).reduce(_ + _)
    })
  }

  def train(observations: Seq[(DenseVector[Double], Double)]) = {

    weights = DenseVector.rand(observations.head._1.length)
    (0 until iterations).foreach(iteration => {
      println("Cost: " + cost(weights, observations))
      println("LR: " + learningRate)
      computeUpdates(weights, observations)
      learningRate = learningRate * ((iterations - Math.log(iteration + 1)) / iterations.toDouble)
     })
  }

  def predict(observation: DenseVector[Double], trueLabel: Option[Double]): (Double, Double) = {
    (sigmoid(weights, observation), trueLabel.getOrElse(Double.MaxValue))
  }

}