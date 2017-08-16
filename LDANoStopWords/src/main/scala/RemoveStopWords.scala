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


object RemoveStopWords {

def main(args: Array[String]) {
	val conf = new SparkConf()
				.setAppName("RemoveStopWords")
				.setMaster("local[*]")
				.set("spark.executor.memory", "2g")
				.set("spark.shuffle.memoryFraction","0.001")
				.set("spark.rdd.compress","true")
    val sc = new SparkContext(conf)

var stopWordsArray = Array("rt","a","an","the","able","about","after","all","almost", "also","am","and","any","are","as","at","be","because","been","but","by", "can","cannot","could","dear","did","do","does","don't","either","else","ever","every","for","from","get","got","had","has","have","he","her","him","his","how","however","i","if","in","into","is", "it","like","may","me","might","most", "must","my","no","nor","not","of","on","only","or","other","our", "own","rather","said","say","she","should","since","so","some","than","that","the","their","them","then","there","these", "they","this","to","too","was","us","was", "we","were","what","when", "where","which","while","who","whom","why","will","with","would","yet","you", "your")

def stopFunc(splitData: Array[String]) : Array[String] = {
var stopArrayBuffer = new ArrayBuffer[String]
for(i<-0 to splitData.length-1) {
if(!(stopWordsArray.contains(splitData(i)))) 
stopArrayBuffer +=splitData(i)
}
var noStopArray = new Array[String](stopArrayBuffer.size)
stopArrayBuffer.copyToArray(noStopArray)
noStopArray
}

val tweetDataStopWords  = sc.textFile("/Users/Alya/Desktop/Final/twit15*.txt")
val twitterStopWordsData  = sc.textFile("/Users/Alya/Desktop/Final/twit15*.txt")


var stopWordstest = tweetDataStopWords.map(line=>line.split("\t")).map(r=>r(1).toLowerCase()).map(data=>data.split(" ")).flatMap(a=>stopFunc(a)).distinct().zipWithIndex

var zippedDataStopWords = stopWordstest.collect

var uniqueWordsStop = stopWordstest.count


def joinFunc(val1: Array[String]) : collection.mutable.Map[Long, org.apache.spark.mllib.linalg.Vector]= {
var throwMap = collection.mutable.Map[Long, org.apache.spark.mllib.linalg.Vector]()
var nonZeroIndices = new Array[Int](val1.length-1) 
var prev = 0
for(i<-0 to val1.length-1) 
{
if(i!=0) 
{
var a = zippedDataStopWords.filter { case(key, value) => key.equals(val1(i).trim) }.map(p=>p._2)
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
throwMap += val1(0).toLong -> Vectors.sparse(uniqueWordsStop.toInt,uniqueNonZero,nonzeroWordCounts)
throwMap
}


var testingStopWords = twitterStopWordsData.map(line=>(line.toLowerCase().split("\t")).flatMap(data=>data.split(" "))).map(a=>stopFunc(a)).flatMap(p=>joinFunc(p)).cache()


val ldaModeltest = new LDA().setK(10).run(testingStopWords)

println("Learned topics (as distributions over vocab of " + ldaModeltest.vocabSize + " words):")

val topics = ldaModeltest.topicsMatrix
for (topicTest <- Range(0, 10)) {
  print("Topic " + topicTest + ":")
  for (word <- Range(0, ldaModeltest.vocabSize)) { print(" " + topics(word, topicTest)); }
  println()
}


def convertToRDD(mat: Matrix): RDD[Vector] = {
  val columns = mat.toArray.grouped(mat.numRows)
  val rows = columns.toSeq.transpose 
  val vectors = rows.map(row => new DenseVector(row.toArray))
  sc.parallelize(vectors)
}



var vec = convertToRDD(topics).collect

val observ = sc.parallelize(vec)
val summary : MultivariateStatisticalSummary = Statistics.colStats(observ)
for(i<-0 to 9)
{
var a = zippedDataStopWords.filter { case(key, value) => value.equals((topics.toArray.indexOf(summary.max(i))-i*topics.numRows).toLong) }.map(p=>p._1)
println(" Top word in Topic "+i+" is "+a(0))
}


}
}


