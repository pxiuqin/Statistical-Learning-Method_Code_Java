package xiuqin.ml.perceptron;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import xiuqin.utils.FileUtils;

//感知机算法实现
public class Perceptron {
    public static void main(String[] args) {
        Perceptron perceptron = new Perceptron();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        String filePath = "data/Mnist/mnist_train.csv";
        System.out.println("read file:" + filePath);
        DataWrapper trainData = perceptron.readData(filePath);
        trainData.rebuildLabel();
        trainData.rebuildData();

        //2、读取测试数据
        filePath = "data/Mnist/mnist_test.csv";
        System.out.println("read file:" + filePath);
        DataWrapper testData = perceptron.readData(filePath);
        testData.rebuildLabel();
        testData.rebuildData();

        //3、训练数据获取权重
        System.out.println("training data");
        perceptron.modelTraining(trainData, 50);

        //4、进行测试并获得准确率
        System.out.println("testing data");
        testData.w = trainData.w;
        testData.b = trainData.b;
        double accuracy = perceptron.modelTest(testData);
        System.out.println("accuracy rate is " + accuracy);

        //5、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);
    }

    //训练数据，包括label和data
    private class DataWrapper {
        INDArray labelArr;
        INDArray dataArr;
        INDArray w;
        float b;

        public INDArray getLabelArr() {
            return labelArr;
        }

        public void setLabelArr(INDArray labelArr) {
            this.labelArr = labelArr;
        }

        public INDArray getDataArr() {
            return dataArr;
        }

        public void setDataArr(INDArray dataArr) {
            this.dataArr = dataArr;
        }

        //Mnsit data have 0-9 lable, two class for 1 of >=5, -1 of <5
        public void rebuildLabel() {
            for (int i = 0; i < labelArr.columns(); i++) {
                labelArr.putScalar(i, labelArr.getFloat(i) >= 5 ? 1 : -1);
            }
        }

        //normal
        public void rebuildData() {
            dataArr = dataArr.div(255.0);
        }
    }

    private DataWrapper readData(String path) {
        DataWrapper data = new DataWrapper();
        String separator = ",";
        try {
            INDArray result = FileUtils.readFromText(path, separator);

            //拆分第一列数据为label
            data.setLabelArr(result.getColumn(0));

            //剩下的数据为data
            int[] cols = new int[(result.columns() - 1)];
            for (int i = 0; i < result.columns() - 1; i++) {
                cols[i] = i + 1;
            }
            data.setDataArr(result.getColumns(cols));
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            return data;
        }
    }

    //perceptron model training
    private void modelTraining(DataWrapper trainData, int iterator) {
        //labelAtr
        INDArray label = trainData.labelArr;

        //create init w
        trainData.w = Nd4j.zeros(trainData.dataArr.columns());

        //create init b
        trainData.b = 0;

        //learning rate
        double h = 0.0001;

        long samples = trainData.dataArr.rows();
        for (int k = 0; k < iterator; k++) {
            for (int i = 0; i < samples; i++) {
                //get current sample
                INDArray xi = trainData.dataArr.getRow(i);

                //get current label
                double yi = label.getDouble(i);

                //SGD
                if (-1 * yi * (trainData.w.mmul(xi.reshape(xi.columns(), 1)).getDouble(0, 0) + trainData.b) >= 0) {
                    trainData.w = trainData.w.add(xi.mul(yi * h));
                    trainData.b += yi * h;
                }
            }

            //print training progressing
            System.out.println(String.format("Round %d:%d", k + 1, iterator));
        }
    }

    //model test
    private double modelTest(DataWrapper testData) {
        //labelAttr
        INDArray label = testData.labelArr;

        //infer error data
        long errorCnt = 0;
        long samples = testData.dataArr.rows();
        for (int i = 0; i < samples; i++) {
            //get current sample
            INDArray xi = testData.dataArr.getRow(i);

            //get current label
            double yi = label.getDouble(i);

            //compute result
            double result = -1 * yi * (testData.w.mmul(xi.reshape(xi.columns(), 1)).getDouble(0, 0) + testData.b);

            //infer error
            if (result >= 0) {
                errorCnt += 1;
            }
        }

        //accuracy=1-(errorCnt/samples)
        return 1 - (1.0 * errorCnt / samples);
    }
}
