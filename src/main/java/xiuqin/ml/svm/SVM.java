package xiuqin.ml.svm;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import xiuqin.ml.ModelBase;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SVM extends ModelBase {
    private double sigma = 10;   //高斯核分母中的σ
    private double C = 200; //惩罚参数
    private double toler = 0.001;     //松弛变量
    private int inter = 100;   //迭代次数
    private double increment = 0.00001;  //a2的改变量阈值
    private double b = 0;    //偏置b
    private List<Integer> supportVecIndex = new ArrayList<>();  //记录支持向量下标

    private long m;  //样本数量
    private INDArray k;        //核函数
    private INDArray alpha;   //α长度为训练集数目
    private INDArray E; //SMO运算过程中的Ei

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
        this.m = this.trainLabelArr.length();
        this.k = calcKernel();
        this.alpha = Nd4j.zeros(this.m);
        this.E = Nd4j.zeros(this.m);
    }

    //使用高斯核计算核函数
    private INDArray calcKernel() {
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
                double xz = calcGaussKernel(X, Z);

                //将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                kernel.putScalar(i, j, xz);
                kernel.putScalar(j, i, xz);
            }
        }

        return kernel;
    }

    //使用高斯核计算核函数
    private double calcGaussKernel(INDArray X, INDArray Z) {
        //按照7.3.3常用的核函数，式7.90计算高斯核
        /**
         * 高斯核计算：exp(-1*(||X-Z||^2)/(2*sigma^2)
         * ||X-Z||^2：2范数的平方理解为向量内积=>(X - Z) * (X - Z).T
         */
        INDArray x_z = X.sub(Z);
        double xz = Transforms.dot(x_z, x_z).getDouble(0);
        xz = Math.exp(-1 * xz / (2 * this.sigma * this.sigma));

        return xz;
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
         * 这一步是一个优化性的算法,见书129页
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
                maxIndex = random.nextInt((int) this.m);
            }
            E2 = calcEi(maxIndex);
        }

        return Pair.create(E2, maxIndex);
    }

    //训练过程
    private void train() {
        int interStep = 0;
        int parameterChanged = 1;

        /**
         * 如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
         * parameterChanged==0时表示上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明
         * 达到了收敛状态，可以停止了
         */
        while (interStep < this.inter && parameterChanged > 0) {
            System.out.println(String.format("iter:%s for %s", interStep, this.inter));
            interStep++;
            parameterChanged = 0;  //新一轮迭代修改标志位

            //大循环遍历所有样本，找到SMO中的第一个变量
            for (int i = 0; i < this.m; i++) {
                //查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if (!isSatisfyKKT(i)) {
                    /**
                     * 如果下标为i的a不满足KKT条件，则进行优化
                     * 第一个变量α的下标i已经确定，接下来按照“7.4.2 变量的选择方法”第二步
                     * 选择变量2。由于变量2的选择中涉及到|E1 - E2|，因此先计算E1
                     */
                    double E1 = calcEi(i);  //计算E1
                    Pair<Double, Integer> E2_j = getAlphaJ(E1, i);
                    double E2 = E2_j.getFirst();
                    int j = E2_j.getSecond();

                    /**
                     * 参考7.4.1两个变量二次规则的求解方法，见书126页
                     */
                    //获取两个变量的标签
                    int y1 = this.trainLabelArr.getInt(i);
                    int y2 = this.trainLabelArr.getInt(j);

                    //复制alpha作为old值
                    double alphaOld1 = this.alpha.getDouble(i);
                    double alphaOld2 = this.alpha.getDouble(j);

                    //依据标签是否一直来生成不同的L和H
                    double L, H;
                    if (y1 != y2) {
                        L = Math.max(0, alphaOld2 - alphaOld1);
                        H = Math.min(this.C, this.C + alphaOld2 - alphaOld1);
                    } else {
                        L = Math.max(0, alphaOld2 + alphaOld1 - this.C);
                        H = Math.min(this.C, alphaOld2 + alphaOld1);
                    }

                    //如果两者相等，说明该变量无法在优化了，直接跳到下一次循环即可
                    if (L == H) continue;

                    /**
                     * 计算alpha的新增，根据7.4.1两个变量二次规划的求解方法公式:7.106更新a2，见书127
                     */
                    //先获取几个K值
                    double k11 = this.k.getDouble(i, i);
                    double k12 = this.k.getDouble(i, j);
                    double k22 = this.k.getDouble(j, j);
                    double k21 = this.k.getDouble(j, i);

                    //根据7.106更新a2，该a2还未经剪切,公式：a2_old+y2(E1-E2)/(k11+k22-2*k12)
                    double alphaNew2 = alphaOld2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12);

                    //剪切a2: a2_new>H:H,L<=a2_new<=H:a2_new,a2_new<L:L
                    if (alphaNew2 < L) {
                        alphaNew2 = L;
                    } else if (alphaNew2 > H) {
                        alphaNew2 = H;
                    }

                    /**
                     * 更新a1，根据7.109
                     * 公式：a1_old+y1*y2*(a2_old-a2_new)
                     */
                    double alphaNew1 = alphaOld1 + y1 * y2 * (alphaOld2 - alphaNew2);

                    /**
                     * 依据“7.4.2 变量的选择方法”第三步式7.115和7.116计算b1和b2,见书130页
                     * 公式：b1=-E1-y1*k11*(a1_new-a1_old)-y2*k21*(a2_new-a2_old)+b_old
                     * 公式：b2=-E2-y1*k12*(a1_new-a1_old)-y2*k22*(a2_new-a2_old)+b_old
                     */
                    double bNew1 = -1 * E1 - y1 * k11 * (alphaNew1 - alphaOld1) - y2 * k21 * (alphaNew2 - alphaOld2) + this.b;
                    double bNew2 = -1 * E2 - y1 * k12 * (alphaNew1 - alphaOld1) - y2 * k22 * (alphaNew2 - alphaOld2) + this.b;

                    //根据a1和a2值的范围确定新b
                    double bNew;
                    if (alphaNew1 > 0 && alphaNew1 < this.C) {
                        bNew = bNew1;
                    } else if (alphaNew2 > 0 && alphaNew2 < this.C) {
                        bNew = bNew2;
                    } else {
                        bNew = (bNew1 + bNew2) / 2;
                    }

                    /**
                     * 将更新后的各类值写入进行更新
                     */
                    this.alpha.putScalar(i, alphaNew1);
                    this.alpha.putScalar(j, alphaNew2);
                    this.b = bNew;

                    this.E.putScalar(i, calcEi(i));
                    this.E.putScalar(j, calcEi(j));

                    /**
                     * 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值，反之则自增
                     */
                    if (Math.abs(alphaNew2 - alphaOld2) >= this.increment) {
                        parameterChanged++;
                    }
                }

                //打印迭代轮数，i值，该迭代轮数修改α数目
                System.out.println(String.format("iter: %d i:%d, pairs changed %d", interStep, i, parameterChanged));
            }
        }

        /**
         * 全部训练结束后，重新遍历一般a，查找里面的支持向量
         */
        for (int i = 0; i < this.m; i++) {
            //如果a>0,说明是支持向量
            if (this.alpha.getDouble(i) > 0) {
                this.supportVecIndex.add(i);
            }
        }
    }

    //预测过程
    private int predict(INDArray x) {
        int result = 0;

        /**
         * 遍历所有支持向量，计算求和式,如果是非支持向量，求和子式必为0，没有必要进行计算
         * SVM最后只有支持向量起作用
         */
        for (Integer index : supportVecIndex) {
            //先单独将核函数计算出来
            double k = this.calcGaussKernel(this.trainDataArr.getRow(index), x);

            //对每一项子式进行求和，最终计算得到求和项的值
            result += this.alpha.getDouble(index) * this.trainLabelArr.getInt(index) * k;
        }

        //求和项计算结束后加上偏置b
        result += this.b;

        if (result > 0) {
            return 1;
        } else if (result < 0) {
            return -1;
        } else {
            return 0;
        }
    }

    //模型测试
    private double modelTest() {
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

    public static void main(String[] args) {
        SVM svm = new SVM();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        System.out.println("loading data");
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
        svm.init();  //加载完数据后进行初始化
        System.out.println("training data");
        svm.train(); //训练数据

        //4、进行测试并获得准确率
        System.out.println("testing data");
        double accuracy = svm.modelTest();
        System.out.println("accuracy rate is " + accuracy);

        //5、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);
    }


}
