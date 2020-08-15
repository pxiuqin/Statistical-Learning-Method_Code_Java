package xiuqin.ml.naivebayes;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.conditions.EqualsCondition;
import org.nd4j.linalg.ops.transforms.Transforms;
import xiuqin.ml.ModelBase;

public class NaiveBayes extends ModelBase {
    INDArray py;  //先验概率
    INDArray px_y;   //类条件概率

    @Override
    public void normalData(double pivot) {
        //大于pivot为1，小于等于为0
        BooleanIndexing.replaceWhere(this.trainDataArr, 0, Conditions.lessThanOrEqual(pivot));
        BooleanIndexing.replaceWhere(this.trainDataArr, 1, Conditions.greaterThan(pivot));

        BooleanIndexing.replaceWhere(this.testDataArr, 0, Conditions.lessThanOrEqual(pivot));
        BooleanIndexing.replaceWhere(this.testDataArr, 1, Conditions.greaterThan(pivot));
    }

    public static void main(String[] args) {
        int labels = 10;
        int features = 784;

        NaiveBayes bayes = new NaiveBayes();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        String filePath = "data/Mnist/mnist_train.csv";
        System.out.println("read file:" + filePath);
        bayes.loadTrainData(filePath, ",");

        //2、读取测试数据
        filePath = "data/Mnist/mnist_test.csv";
        System.out.println("read file:" + filePath);
        bayes.loadTestData(filePath, ",");
        bayes.normalData(128);   //normalData,这里按照样本大小的数值来分成两类，大于128为1，小于等于为0，值最大为255

        //3、训练模型
        System.out.println("training data");
        bayes.getAllProbability(labels, features);

        //4、进行测试并获得准确率
        System.out.println("testing data");
        double accuracy = bayes.modelTest(labels, features);
        System.out.println("accuracy rate is " + accuracy);

        //5、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);
    }

    //获取所有样本先验概率和类条件概率
    private void getAllProbability(int labels, int features) {
        this.py = Nd4j.zeros(labels);  //初始化先验概率为0

        /**
         * 这部分合并到计算P(x|y)中了
        //找出每种标签的先验概率
        for (int i = 0; i < labels; i++) {
            //INDArray temp = this.trainLabelArr.eq(i).castTo(DataType.INT8);
            //int count=temp.sumNumber().doubleValue();

            *//*int count = 0;
            for (int j = 0; j < this.trainLabelArr.columns(); j++) {
                if (this.trainLabelArr.getInt(j) == i) {
                    count += 1;
                }
            }
            double prob = 1.0 * count / this.trainLabelArr.columns();  //给定常数是为了应对空值*//*

            double prob = this.trainLabelArr.scan(new EqualsCondition(i)).doubleValue();
            this.py.putScalar(i, prob);
        }

        this.py = Transforms.log(this.py);  //log处理
        */

        //初始shape和全0，这里2表示初始把x数据分为0和1两类
        this.px_y = Nd4j.zeros(labels, features, 2);

        //完成给标签的累加和
        for (int i = 0; i < this.trainLabelArr.columns(); i++) {
            int label = this.trainLabelArr.getInt(i);  //找到当前label
            INDArray sample = this.trainDataArr.getRow(i);  //获取当前要处理的样本

            //对样本的每一个维度特征进行遍历，完成确定标签下的累加和
            for (int j = 0; j < features; j++) {
                int[] index = new int[]{label, j, sample.getInt(j)};

                //累加确定标签下的样本的累加和
                this.px_y.putScalar(index, this.px_y.getInt(index) + 1);  //累加
            }
        }

        //计算条件概率 Px_y=P(X=x|Y=y)
        for (int i = 0; i < labels; i++) {
            for (int j = 0; j < features; j++) {
                //获取y=label，第j个特征为0的个数
                int px_y_0 = this.px_y.getInt(i, j, 0);

                //获取y=label，第j个特征为1的个数
                int px_y_1 = this.px_y.getInt(i, j, 1);

                //分别计算对于y= label，x第j个特征为0和1的条件概率分布
                this.px_y.putScalar(new int[]{i, j, 0}, Math.log(1.0 * (px_y_0 + 1) / (px_y_0 + px_y_1 + 2)));
                this.px_y.putScalar(new int[]{i, j, 1}, Math.log(1.0 * (px_y_1 + 1) / (px_y_0 + px_y_1 + 2)));
            }

            //计算P(y)
            double prob = this.trainLabelArr.scan(new EqualsCondition(i)).doubleValue();
            this.py.putScalar(i, Math.log(prob));  //直接log处理了
        }
    }

    //获取最大概率
    private int getArgMax(int labels, int features, INDArray sample) {
        //nd4j中有argMax函数，是否能直接使用

        //建立存放所有标记的估计概率数组
        INDArray prob = Nd4j.zeros(labels);

        //对于每一个类别，单独估计其概率
        for (int i = 0; i < labels; i++) {
            //初始化sum为0，sum为求和项。
            //在训练过程中对概率进行了log处理，所以这里原先应当是连乘所有概率，最后比较哪个概率最大,但是当使用log处理时，连乘变成了累加，所以使用sum
            double sum = 0;
            for (int j = 0; j < features; j++) {
                sum += this.px_y.getDouble(i, j, sample.getInt(j));  //用累乘时值会无线趋近0，从而影响确定类标记
            }

            //最后再和先验概率相加（也就是式4.7中的先验概率乘以后头那些东西，乘法因为log全变成了加法）
            prob.putScalar(i, sum + this.py.getDouble(i));
        }

        //找到该概率最大值对应的所有（索引值和标签值相等）
        /*return BooleanIndexing
                .firstIndex(prob, new EqualsCondition(Nd4j.max(prob).getDouble(0)))
                .getInt(0);*/
        return Nd4j.argMax(prob).getInt(0);
    }

    private double modelTest(int labels, int features) {
        int errorCount = 0;

        int testCount = this.testDataArr.rows();
        for (int i = 0; i < testCount; i++) {
            INDArray each = this.testDataArr.getRow(i);
            long label = getArgMax(labels, features, each);

            if (label != this.testLabelArr.getInt(i)) {
                errorCount += 1;
            }

            if (i % 500 == 0) {
                System.out.println("testing:" + i);
            }
        }

        return 1 - 1.0 * errorCount / testCount;
    }
}
