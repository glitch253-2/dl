import breeze.linalg.DenseVector

/**
 * Created by Mike on 1/14/2015.
 */
trait Classifier {

  def train(Seq[(DenseVector[Double], Double)])

  def predict(DenseVector[Double], Option[Double])

}
