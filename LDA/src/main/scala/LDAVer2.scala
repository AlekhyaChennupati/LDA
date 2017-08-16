import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.collection
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}


object LDAVer2 {

def main(args: Array[String]) {
	val conf = new SparkConf()
				.setAppName("LDA Ver2")
				.setMaster("local[*]")
				.set("spark.executor.memory", "2g")
				.set("spark.shuffle.memoryFraction","0.001")
				.set("spark.rdd.compress","true")
    val sc = new SparkContext(conf)

val tweetData = sc.textFile("/Users/Alya/Desktop/Final/twit25.txt")
val twitterData = sc.textFile("/Users/Alya/Desktop/Final/twit25.txt")

var test = tweetData.map(line=>line.split("\t")).map(r=>r(1).toLowerCase()).flatMap(data=>data.split(" ")).distinct().zipWithIndex

var zippedData = test.collect

var uniqueWords = test.count

println("**********************Unique Words = "+uniqueWords+" **************************")

def joinFunc(val1: Array[String]) : collection.mutable.Map[Long, org.apache.spark.mllib.linalg.Vector]= {
var throwMap = collection.mutable.Map[Long, org.apache.spark.mllib.linalg.Vector]()
var nonZeroIndices = new Array[Int](val1.length-1) 
var prev = 0
for(i<-0 to val1.length-1) 
{
if(i!=0) 
{
var a = zippedData.filter { case(key, value) => key.equals(val1(i).trim) }.map(p=>p._2)
nonZeroIndices(i-1) = a(0).toInt
}
}
scala.util.Sorting.quickSort(nonZeroIndices)
var uniqueNonZero  = new Array[Int](nonZeroIndices.distinct.length)
var nonzeroWordCounts = new Array[Double](nonZeroIndices.distinct.length)
nonZeroIndices.distinct.copyToArray(uniqueNonZero)
var occur = 0
var temp = -1
var k = -1
for(j<-0 to nonZeroIndices.length-1) {
if(temp == nonZeroIndices(j))
{
occur = occur + 1
nonzeroWordCounts(k) = occur
}
else {
k = k+1
occur=1
nonzeroWordCounts(k) = occur
temp = nonZeroIndices(j)
}
}
throwMap += val1(0).toLong -> Vectors.sparse(uniqueWords.toInt,uniqueNonZero,nonzeroWordCounts)
throwMap
}

var testing = twitterData.map(line=>(line.toLowerCase().split("\t")).flatMap(data=>data.split(" "))).flatMap(p=>joinFunc(p)).cache()


val ldaModeltest = new LDA().setK(10).run(testing)

println("Learned topics (as distributions over vocab of " + ldaModeltest.vocabSize + " words):")

val topics = ldaModeltest.topicsMatrix
/*for (topicTest <- Range(0, 10)) {
  print("Topic " + topicTest + ":")
  for (word <- Range(0, ldaModeltest.vocabSize)) { print(" " + topics(word, topicTest)); }
  println()
}*/


def convertToRDD(mat: Matrix): RDD[Vector] = {
  val columns = mat.toArray.grouped(mat.numRows)
  val rows = columns.toSeq.transpose 
  val vectors = rows.map(row => new DenseVector(row.toArray))
  sc.parallelize(vectors)
}
var vec = convertToRDD(topics).collect
val observ = sc.parallelize(vec)
val summary : MultivariateStatisticalSummary = Statistics.colStats(observ)
for(i<- 0 to 9)
{
var a = zippedData.filter { case(key, value) => value.equals((topics.toArray.indexOf(summary.max(i))-i*topics.numRows).toLong) }.map(p=>p._1)
println(" Top word in Topic "+i+" is "+a(0))
}
}
}