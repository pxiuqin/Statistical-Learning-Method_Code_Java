package xiuqin.ml.knn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.EqualsCondition;
import xiuqin.ml.ModelBase;

public class KNN4KDTree extends ModelBase {
    KDTree kd;

    public static void main(String[] args) {
        int topK = 25;
        int labels = 10;

        KNN4KDTree knn = new KNN4KDTree();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        String filePath = "data/Mnist/mnist_train.csv";
        System.out.println("read file:" + filePath);
        knn.loadTrainData(filePath, ",");

        //2、读取测试数据
        filePath = "data/Mnist/mnist_test.csv";
        System.out.println("read file:" + filePath);
        knn.loadTestData(filePath, ",");

        //3、生成KDTree
        System.out.println("create kdtree");
        knn.normalData(255.0f); //  归一化数据，在创建kdtree是我们构建一个0到1的区间边界
        knn.createKDTree();

        //4、进行测试并获得准确率
        System.out.println("training data");
        double accuracy = knn.modelTest(topK, labels);
        System.out.println("accuracy rate is " + accuracy);

        //5、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);
    }

    //创建KDTree
    private void createKDTree() {
        double[][] ps = this.trainDataArr.toDoubleMatrix();
        int num = ps.length;
        HyperPoint[] hps = new HyperPoint[num];
        for (int i = 0; i < num; i++)
            hps[i] = new HyperPoint(ps[i]);

        HyperPoint min = new HyperPoint(Nd4j.zeros(this.trainDataArr.columns()).toDoubleVector());  //构建一个全0的最小边界
        HyperPoint max = new HyperPoint(Nd4j.ones(this.trainDataArr.columns()).toDoubleVector());   //构建一个全1的最大边界
        int K = ps[0].length;

        this.kd = new KDTree(K, min, max);
        //this.kd.insertByMedianFinding(hps);

         for (int i = 0; i < num; i++) {
             kd.insert(hps[i]);
         }
    }

    /**
     * get closest sample label
     *
     * @param sample test sample
     * @param topK   topK
     * @return label
     */
    private long getClosest(INDArray sample, int topK) {
        double[] vector = sample.toDoubleVector();
        HyperPoint hp = new HyperPoint(vector);
        HyperPoint nearestPoint = kd.nearestPoint(hp);

        int index = 0;
        for (int i = 0; i < this.trainDataArr.rows(); i++) {
            double[] each = this.trainDataArr.getRow(i).toDoubleVector();
            if (nearestPoint.coords==each) {
                index = i;
                break;
            }
        }
        
        //return label
        return this.trainLabelArr.getLong(index);
    }

    private double modelTest(int topK, int labels) {
        int errorCount = 0;

        int testCount = 20;    //this.testDataArr.rows();
        for (int i = 0; i < testCount; i++) {
            INDArray each = this.testDataArr.getRow(i);
            long label = getClosest(each, topK);

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
