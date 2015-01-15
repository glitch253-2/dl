import breeze.linalg.DenseVector

/**
 * Created by Mike on 1/14/2015.
 */
trait Classifier {

  def train(observations: Seq[(DenseVector[Double], Double)])
  def predict(instance: DenseVector[Double], trueResponse: Option[Double]): (Double, Double)

}
