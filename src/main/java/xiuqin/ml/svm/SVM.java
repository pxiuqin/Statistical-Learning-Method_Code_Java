package xiuqin.ml.svm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import xiuqin.ml.ModelBase;

public class SVM extends ModelBase {
    private double sigma;   //高斯核分母中的σ
    private double C; //惩罚参数
    private double toler;     //松弛变量

    private INDArray k;        //核函数
    private double b;    //偏置b
    private INDArray alpha;   //α，长度为训练集数目
    private INDArray E; //SMO运算过程中的Ei
    private INDArray supportVecIndex;

    //Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为-1
    @Override
    protected void normalLabel(float pivot) {
        BooleanIndexing.replaceWhere(trainLabelArr, 1, Conditions.equals(pivot));
        BooleanIndexing.replaceWhere(trainLabelArr, -1, Conditions.greaterThan(pivot));

        BooleanIndexing.replaceWhere(testLabelArr, 1, Conditions.equals(pivot));
        BooleanIndexing.replaceWhere(testLabelArr, -1, Conditions.greaterThan(pivot));
    }

    //初始化处理
    private void init() {

    }

    //使用高斯核计算核函数
    private INDArray calcKernel() {
        int m = this.trainDataArr.rows();
        INDArray kernel = Nd4j.zeros(m, m);  //初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m

        //循环遍历每个样本
        for (int i = 0; i < m; i++) {
            if (i % 100 == 0) System.out.println("construct the kernel:" + i);

            //获取X训练样本
            INDArray X = this.trainDataArr.getRow(i);

            /**
             * 小循环遍历Xj，Xj为式7.90中的Z
             * 由于 Xi * Xj 等于 Xj * Xi，一次计算得到的结果可以
             * 同时放在k[i][j]和k[j][i]中，这样一个矩阵只需要计算一半即可
             * 所以小循环直接从i开始
             */
            for (int j = i; j < m; j++) {
                //获取的Z训练样本
                INDArray Z = this.trainDataArr.getRow(j);

                /**
                 * 高斯核计算：exp(-1*(||X-Z||^2)/(2*sigma^2)
                 * ||X-Z||^2：2范数的平方理解为向量内积=>(X - Z) * (X - Z).T
                 */
                INDArray x_z = X.subi(Z);
                double xz = Transforms.dot(x_z, x_z).getDouble(0);
                xz = Math.exp(-1 * xz / (2 * this.sigma * this.sigma));

                //将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k.putScalar(i, j, xz);
                k.putScalar(j, i, xz);
            }
        }

        return k;
    }

    public static void main(String[] args) {
        SVM svm = new SVM();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        String filePath = "data/Mnist/mnist_train.csv";
        System.out.println("read file:" + filePath);
        svm.loadTrainData(filePath, ",");

        //2、读取测试数据
        filePath = "data/Mnist/mnist_test.csv";
        System.out.println("read file:" + filePath);
        svm.loadTestData(filePath, ",");
        svm.normalLabel(0);
        svm.normalData(255);  //正则化处理下数据

        //3、训练
        System.out.println("training data");

        //4、进行测试并获得准确率
        System.out.println("testing data");
        double accuracy = 1; //svm.modelTest();
        System.out.println("accuracy rate is " + accuracy);

        //5、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);
    }


}
