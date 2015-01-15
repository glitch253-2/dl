import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Created by Mike on 1/14/2015.
 */
trait Classifier {

  def train(observations: DenseMatrix[Double], labels: DenseVector[Double])
  def predict(instance: DenseVector[Double], trueResponse: Option[Double]): (Double, Double)

}
