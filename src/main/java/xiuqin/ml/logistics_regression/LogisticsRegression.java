package xiuqin.ml.logistics_regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import xiuqin.ml.ModelBase;

public class LogisticsRegression extends ModelBase {
    INDArray w;  //model params

    public static void main(String[] args) {
        LogisticsRegression lr = new LogisticsRegression();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        String filePath = "data/Mnist/mnist_train.csv";
        System.out.println("read file:" + filePath);
        lr.loadTrainData(filePath, ",");

        //2、读取测试数据
        filePath = "data/Mnist/mnist_test.csv";
        System.out.println("read file:" + filePath);
        lr.loadTestData(filePath, ",");
        lr.normalLabel(5);
        lr.normalData(255);  //正则化处理下数据

        //3、训练
        System.out.println("training data");
        lr.logisticsRegression(20);  //set iteration

        //4、进行测试并获得准确率
        System.out.println("testing data");
        double accuracy = lr.modelTest();
        System.out.println("accuracy rate is " + accuracy);

        //5、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);
    }

    private int predict(INDArray sample) {
        INDArray wx = Transforms.dot(this.w, sample);

        double p1 = Transforms.sigmoid(wx).getDouble(0);

        if (p1 >= 0.5) {
            return 1;
        } else {
            return 0;
        }
    }

    private void logisticsRegression(int iteration) {
        //add 1 to trainData,because of combing w and b
        this.trainDataArr = Nd4j.hstack(this.trainDataArr, Nd4j.ones(this.trainDataArr.rows(), 'f'));

        //init w
        this.w = Nd4j.zeros(this.trainDataArr.columns());

        double h = 0.001;  //set step

        for (int i = 0; i < iteration; i++) {
            System.out.println("training number of iterations is " + (i + 1));

            for (int j = 0; j < this.trainDataArr.rows(); j++) {
                INDArray xi = this.trainDataArr.getRow(j);
                int yi = this.trainLabelArr.getInt(j);
                double wx = Transforms.dot(this.w, xi).getDouble(0);

                //1.通过对数似然函数得：sum(yi*(w*xi)-log(1+exp(w*xi)))
                //2.然后基于上述对数似然求导得：yi*xi-(xi*exp(w*xi)/(1+exp(w*xi)))
                //3.使用梯度上升处理,可以理解成交叉熵的梯度[sum(xi*(yi-p(xi))]，当模型输出概率偏离于真实概率时，梯度较大，加快训练速度，当拟合值接近于真实概率时训练速度变缓慢
                INDArray wi = xi.mul(yi).sub(xi.mul(Math.log(wx)/(1 - Math.log(wx))));
                this.w = this.w.add(wi.mul(h));
            }
        }
    }

    private double modelTest() {
        //testDataArr the same trainDataArr
        this.testDataArr = Nd4j.hstack(this.testDataArr, Nd4j.ones(this.testDataArr.rows(), 'f'));

        int errorCount = 0;

        int testCount = this.testDataArr.rows();
        for (int i = 0; i < testCount; i++) {
            INDArray each = this.testDataArr.getRow(i);
            long label = predict(each);   //get predict

            if (label != this.testLabelArr.getLong(i)) {
                errorCount += 1;
            }

            if (i % 500 == 0) {
                System.out.println("testing:" + i);
            }
        }

        return 1 - 1.0 * errorCount / testCount;
    }
}
