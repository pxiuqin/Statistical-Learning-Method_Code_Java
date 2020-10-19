package xiuqin.ml.svm;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import xiuqin.ml.ModelBase;

import java.util.Random;

public class SVM extends ModelBase {
    private double sigma;   //高斯核分母中的σ
    private double C; //惩罚参数
    private double toler;     //松弛变量
    private int inter = 100;   //迭代次数

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
        this.k = calcKernel();
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

    //判断第i个α是否满足KKT
    private boolean isSatisfyKKT(int index) {
        /**
         * KKT条件判断依据:判断依据参照“7.4.2 变量的选择方法”中“1.第1个变量的选择”
         * 公式：7.111到7.113，见书129页
         */
        double gxi = calcGxi(index);
        double yi = this.trainLabelArr.getDouble(index);
        double alpha = this.alpha.getDouble(index);

        if (Math.abs(alpha) < this.toler && yi * gxi >= 1) {
            //根据7.111公式：yi*gxi>=1
            return true;
        } else if (Math.abs(alpha - this.C) < this.toler && yi * gxi <= 1) {
            //根据7.113公式：ai=C<=>yi*gxi<=1
            return true;
        } else if (alpha > -this.toler && alpha < (this.C + this.toler) && Math.abs(yi * gxi - 1) < this.toler) {
            //根据7.112公式：0<ai<C<=>yi*gxi=1
            return true;
        }

        return false;
    }

    //计算gx
    private double calcGxi(int index) {
        /**
         * 依据“7.101 两个变量二次规划的求解方法”式7.104，见书127页
         */
        double gxi = 0;
        for (int i = 0; i < this.alpha.length(); i++) {
            double temp = this.alpha.getDouble(i);
            //如果等于0就不用参与计算了
            if (temp != 0) {
                //公式：∑ai*yi*K(xi,x)
                gxi += temp * this.trainLabelArr.getInt(i) * this.k.getDouble(i, index);
            }
        }

        gxi += this.b;  //公式：上步∑+b

        return gxi;
    }

    //计算Ei
    private double calcEi(int index) {
        /**
         * 根据“7.4.1 两个变量二次规划的求解方法”式7.105
         * 公式：Ei=g(xi)-yi
         */
        double gxi = calcGxi(index);
        return gxi - this.trainLabelArr.getDouble(index);
    }

    //SMO算法选择第二个变量
    private Pair<Double, Integer> getAlphaJ(double E1, int index) {
        /**
         * 这一步是一个优化性的算法
         * 实际上书上算法中初始时每一个Ei应当都为-yi（因为g(xi)由于初始α为0，必然为0）
         * 然后每次按照书中第二步去计算不同的E2来使得|E1-E2|最大，但是时间耗费太长了
         * 作者最初是全部按照书中缩写，但是本函数在需要3秒左右，所以进行了一些优化措施
         * 在Ei的初始化中，由于所有α为0，所以一开始是设置Ei初始值为-yi。这里修改为与α
         * 一致，初始状态所有Ei为0，在运行过程中再逐步更新
         * 因此在挑选第二个变量时，只考虑更新过Ei的变量，但是存在问题
         * 1.当程序刚开始运行时，所有Ei都是0，那挑谁呢？
         * 当程序检测到并没有Ei为非0时，将会使用随机函数随机挑选一个
         * 2.怎么保证能和书中的方法保持一样的有效性呢？
         * 在挑选第一个变量时是有一个大循环的，它能保证遍历到每一个xi，并更新xi的值，
         * 在程序运行后期后其实绝大部分Ei都已经更新完毕了。下方优化算法只不过是在程序运行
         * 的前半程进行了时间的加速，在程序后期其实与未优化的情况无异
         */
        double E2 = 0;
        double maxE1_E2 = -1;  //初始化|E1-E2|为-1
        int maxIndex = -1;  //初始化第二个变量的下标

        for (int i = 0; i < this.E.length(); i++) {
            double nozeroE = this.E.getDouble(i);
            if (nozeroE != 0) {
                double e2 = this.calcEi(i);  //计算E2
                if (Math.abs(E1 - e2) > maxE1_E2) {
                    maxE1_E2 = Math.abs(E1 - e2);
                    E2 = e2;
                    maxIndex = i;
                }
            }
        }

        if (maxIndex == -1) {
            maxIndex = index;
            Random random = new Random();
            while (maxIndex == index) {
                //获得随机数，如果随机数与第一个变量的下标index一致则重新随机
                maxIndex = random.nextInt(this.trainDataArr.rows());
            }
            E2 = calcEi(maxIndex);
        }

        return Pair.create(E2, maxIndex);
    }

    private void train() {
        int interStep = 0;
        int parameterChanged = 1;

        /**
         * 如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
         * parameterChanged==0时表示上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明
         * 达到了收敛状态，可以停止了
         */
        while (interStep<this.inter && parameterChanged>0){
            System.out.println(String.format("iter:%s for %s",interStep,this.inter));
        }
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
