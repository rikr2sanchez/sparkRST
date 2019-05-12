import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.LongType

object SparkRST {

  def main(args: Array[String]): Unit = {

    //Optionally: turn off logger, so prints are more clear
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .appName("SparkRST")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    val data = spark.read.csv("data/test.csv")

    val indexed = zipWithIndex(data)
    indexed.cache()
    val cols = data.columns.reverse.tail.reverse
    val data1 = cols.
      foldLeft(indexed)( (accDF, c) =>
        accDF.withColumn(c, col(c))
      ).
      select(col("index").alias("object"), array(cols.map(col): _*).as("features"))

    val dt = data1
      .select(col("features").as("feat"),col("object"))
      .groupBy(col("feat"))
      .agg(collect_set(col("object")).alias("objects"))


    val colsAll = data.columns

    val dataAll = colsAll.
      foldLeft(indexed)( (accDF, c) =>
        accDF.withColumn(c, col(c))
      ).
      select(col("index").alias("object"), array(colsAll.map(col): _*).as("features"))

    val dtAll = dataAll
      .select(col("features"),col("object"))
      .groupBy(col("features"))
      .agg(collect_set(col("object")).alias("objects"))

    def myExtract[T](x: Seq[T], i: Int) = x(i)
    // define UDF for extracting strings
    val extractString = udf(myExtract[String] _)

    def segExtract[T](x:Seq[T], i0: Int, i1: Int) = x.slice(i0,i1)
    val extractStringSeg = udf(segExtract[String] _)

    val dt3 = dtAll
        .withColumn("class", extractString(col("features"), lit(colsAll.length-1)))
        .withColumn("feat", extractStringSeg(col("features"), lit(0),lit(colsAll.length-1)))


    val dtA = dt.join(dt3,usingColumn = "feat")

    val c = dtA.select(col("class")).distinct().collect().map(row => row.getAs[String](0))

    val df1 = dtA
      .filter(col("class")===c(0)).select(dt.col("objects"))

    val df2 = dtA
      .filter(col("class")===c(1)).select(dt.col("objects"))


    val dfx = df1.intersect(df2)
    val df3 = df1.union(df2).except(dfx)

    println("Lower Aprox " + c(1))
    df3.intersect(df2).show()

    println("Lower Aprox " + c(0))
    df3.intersect(df1).show()

    println("Upper aprox")
    dt.join(dt3,usingColumn = "feat")
      .select(dt.col("objects"),col("class"))
      .groupBy(col("class"))
      .agg(collect_list(col("objects")))
      .show(truncate=false)

  }

  def zipWithIndex(df: DataFrame, offset: Long = 1, indexName: String = "index") = {
    val dfWithPartitionId = df.withColumn("partition_id", spark_partition_id()).withColumn("inc_id", monotonically_increasing_id())

    val partitionOffsets = dfWithPartitionId
      .groupBy("partition_id")
      .agg(count(lit(1)) as "cnt", first("inc_id") as "inc_id")
      .orderBy("partition_id")
      .select(sum("cnt").over(Window.orderBy("partition_id")) - col("cnt") - col("inc_id") + lit(offset) as "cnt" )
      .collect()
      .map(_.getLong(0))
      .toArray

    dfWithPartitionId
      .withColumn("partition_offset", udf((partitionId: Int) => partitionOffsets(partitionId), LongType)(col("partition_id")))
      .withColumn(indexName, col("partition_offset") + col("inc_id"))
      .drop("partition_id", "partition_offset", "inc_id")
  }
}
