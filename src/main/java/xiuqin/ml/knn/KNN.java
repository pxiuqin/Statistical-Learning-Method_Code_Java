package xiuqin.ml.knn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.EqualsCondition;
import xiuqin.ml.ModelBase;

public class KNN extends ModelBase {

    public static void main(String[] args) {
        int topK = 25;
        int labels = 10;

        KNN knn = new KNN();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        String filePath = "data/Mnist/mnist_train.csv";
        System.out.println("read file:" + filePath);
        knn.loadTrainData(filePath, ",");

        //2、读取测试数据
        filePath = "data/Mnist/mnist_test.csv";
        System.out.println("read file:" + filePath);
        knn.loadTestData(filePath, ",");

        //3、进行测试并获得准确率
        System.out.println("training data");
        double accuracy = knn.modelTest(topK, labels);
        System.out.println("accuracy rate is " + accuracy);

        //4、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);

    }

    private double calcDistance(INDArray x1, INDArray x2) {
        return x1.distance2(x2); //EuclideanDistance
    }

    /**
     * get closest sample label
     *
     * @param sample test sample
     * @param topK   topK
     * @param labels count of label
     * @return label
     */
    private long getClosest(INDArray sample, int topK, int labels) {
        //distance of one sample to training sample
        INDArray distArray = Nd4j.zeros(this.trainDataArr.rows());

        for (int i = 0; i < this.trainDataArr.rows(); i++) {
            INDArray each = this.trainDataArr.getRow(i);

            //calc distance
            double dist = calcDistance(sample, each);

            //add distList
            distArray.putScalar(i, dist);
        }

        //sort the distList
        INDArray topKArray = distArray.dup();
        Nd4j.sort(topKArray, true);

        //create a labelList to store the number of votes
        INDArray labelArray = Nd4j.zeros(labels);

        //voting
        for (int i = 0; i < topK; i++) {
            double dist = topKArray.getDouble(i);  //get dist value
            int index = BooleanIndexing.firstIndex(distArray, new EqualsCondition(dist)).getInt(0);  //get topK index
            int label = this.trainLabelArr.getInt(index);  //trans index to label
            labelArray.putScalar(label, labelArray.getInt(label) + 1);  //votes accumulate
        }

        //return max vote label
        return BooleanIndexing
                .firstIndex(labelArray, new EqualsCondition(Nd4j.max(labelArray).getInt(0)))
                .getInt(0);
    }

    private double modelTest(int topK, int labels) {
        int errorCount = 0;

        int testCount = 200;    //this.testDataArr.rows();
        for (int i = 0; i < testCount; i++) {
            INDArray each = this.testDataArr.getRow(i);
            long label = getClosest(each, topK, labels);

            if (label != this.testLabelArr.getLong(i)) {
                errorCount += 1;
            }

            if (i % 10 == 0) {
                System.out.println("testing:" + i);
            }
        }

        return 1 - 1.0 * errorCount / testCount;
    }
}
