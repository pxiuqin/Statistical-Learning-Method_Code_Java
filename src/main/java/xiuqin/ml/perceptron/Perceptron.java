package xiuqin.ml.perceptron;

import org.nd4j.linalg.api.ndarray.INDArray;
import xiuqin.utils.FileUtils;

//感知机算法实现
public class Perceptron {
    public static void main(String[] args) {
        Perceptron perceptron=new Perceptron();

        //1、读取训练数据
        String filePath = "D:\\codes\\opensource\\ML\\Statistical-Learning-Method_Code_Java\\data\\Mnist\\mnist_train.csv";
        TrainData trainData = perceptron.readTrainingData(filePath);

        //2、读取测试数据


        //3、训练数据获取权重


        //4、进行测试并获得准确率


        //5、计算所用时间
    }

    //训练数据，包括label和data
    private class TrainData {
        INDArray labelArr;
        INDArray dataArr;

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
    }

    private TrainData readTrainingData(String path) {
        TrainData trainData = new TrainData();
        String separator = ",";
        try {
            INDArray result = FileUtils.readFromText(path, separator);

            //拆分第一列数据为label
            trainData.setLabelArr(result.getColumn(0));

            //剩下的数据为data
            trainData.setDataArr(result.getColumns(1,23));
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            return trainData;
        }
    }
}
