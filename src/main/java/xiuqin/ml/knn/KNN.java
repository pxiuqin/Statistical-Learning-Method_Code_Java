package xiuqin.ml.knn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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

        //5、计算所用时间
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
        INDArray distList = Nd4j.zeros(this.trainDataArr.rows());

        for (int i = 0; i < this.trainDataArr.rows(); i++) {
            INDArray each = this.trainDataArr.getRow(i);

            //calc distance
            double dist = calcDistance(sample, each);

            //add distList
            distList.putScalar(i, dist);
        }

        //sort the distList
        int[] cols = new int[topK];
        for (int i = 0; i < topK; i++) {
            cols[i] = i;
        }
        INDArray topKList = Nd4j.sort(distList, false).getScalar(cols);

        //create a labelList to store the number of votes
        INDArray labelList = Nd4j.zeros(labels);

        //voting
        for (int i = 0; i < topKList.columns(); i++) {
            int index = topKList.getInt(i);  //get topK index
            int label = this.trainLabelArr.getInt(index);  //trans index to label
            labelList.putScalar(label, labelList.getInt(label) + 1);  //votes accumulate
        }
        return 5;
        //return max vote label
        //return labelList.linearIndex(Nd4j.max(labelList).getInt(0));
    }

    private double modelTest(int topK, int labels) {
        int errorCount = 0;

        int testCount = this.testDataArr.rows();
        for (int i = 0; i < testCount; i++) {
            INDArray each = this.testDataArr.getRow(i);
            long label = getClosest(each, topK, labels);

            if (label != this.testLabelArr.getLong(i)) {
                errorCount += 1;
            }
        }

        return 1 - 1.0 * errorCount / testCount;
    }
}
