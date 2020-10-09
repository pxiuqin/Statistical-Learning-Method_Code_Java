package xiuqin.ml.maximum_entropy;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.conditions.EqualsCondition;
import xiuqin.ml.ModelBase;

import java.util.HashMap;
import java.util.Map;

public class MaximumEntropy extends ModelBase {
    INDArray w;   //Pw(y|x)中的w
    int n;   //训练集中（xi，y）对数量
    Map<Integer, IdFxy> fxy = new HashMap<>();  //所有(x, y)对出现的次数
    Map<String, Integer> xy2index = new HashMap<>();   //把所有idfxy打平转换成有指针
    Map<Integer, String> index2xy = new HashMap<>();   //相反操作通过index获取xy
    INDArray ep_xy;   //Ep_xy期望值
    int iterations = 5;  //迭代次数
    int M=10000;  //IIS算法中使用的f#(x,y)为常数，见书91页

    //Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为0
    @Override
    protected void normalLabel(float pivot) {
        BooleanIndexing.replaceWhere(trainLabelArr, 1, Conditions.equals(pivot));
        BooleanIndexing.replaceWhere(trainLabelArr, 0, Conditions.greaterThan(pivot));

        BooleanIndexing.replaceWhere(testLabelArr, 1, Conditions.equals(pivot));
        BooleanIndexing.replaceWhere(testLabelArr, 0, Conditions.greaterThan(pivot));
    }

    @Override
    protected void normalData(double pivot) {
        BooleanIndexing.replaceWhere(trainLabelArr, 0, Conditions.lessThanOrEqual(pivot));
        BooleanIndexing.replaceWhere(trainLabelArr, 1, Conditions.greaterThan(pivot));

        BooleanIndexing.replaceWhere(testLabelArr, 0, Conditions.lessThanOrEqual(pivot));
        BooleanIndexing.replaceWhere(testLabelArr, 1, Conditions.greaterThan(pivot));
    }

    public static void main(String[] args) {
        MaximumEntropy me = new MaximumEntropy();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        String filePath = "data/Mnist/mnist_train.csv";
        System.out.println("read file:" + filePath);
        me.loadTrainData(filePath, ",");

        //2、读取测试数据
        filePath = "data/Mnist/mnist_test.csv";
        System.out.println("read file:" + filePath);
        me.loadTestData(filePath, ",");
        me.normalLabel(0);
        me.normalData(128);  //正则化处理下数据

        //3、训练
        me.init();
        System.out.println("training data");
        me.maxEntropyTrain();

        //4、进行测试并获得准确率
        System.out.println("testing data");
        double accuracy = me.modelTest();
        System.out.println("accuracy rate is " + accuracy);

        //5、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);
    }

    //计算特征函数f(x, y)关于模型P(Y|X)与经验分布P_(X)的期望值（P后带下划线“_”表示P上方的波浪线【见书82-83页】
    private INDArray calcEpxy() {
        INDArray epxy = Nd4j.zeros(this.n);  //条件概率期望的个数

        for (int i = 0; i < this.trainDataArr.rows(); i++) {
            INDArray row = this.trainDataArr.getRow(i);

            //初始化公式中的P(y|x)列表
            double[] pwxy = new double[2];  //因为特征函数给定的是0和1

            //计算P(y = 0 | X),注：程序中X表示是一个样本的全部特征，x表示单个特征，这里是全部特征的一个样本
            pwxy[0] = calcPwy_x(row, 0);

            //计算P(y = 1 | X)
            pwxy[1] = calcPwy_x(row, 1);

            //针对一个训练样本，处理每个条件概率期望
            for (int j = 0; j < row.columns(); j++) {
                for (int k = 0; k < 2; k++) {
                    String fi_x_y = String.format("%s_%s_%s", j, row.getInt(j), k);  //这里k做了简化，因为label处理成0和1了
                    if (this.xy2index.containsKey(fi_x_y)) {
                        //在xy->index字典中指定当前特征i，以及(x, y)对：(xi, y)，读取其index
                        int index = this.xy2index.get(fi_x_y);

                        //计算每种特征的概率[P(x)*P(y|x)]
                        double prob = epxy.getDouble(index) + (1.0 / this.trainDataArr.rows()) * pwxy[k];
                        epxy.putScalar(index, prob);
                    }
                }
            }
        }

        return epxy;
    }

    //计算特征函数f(x, y)关于经验分布P_(X,Y)的期望值（P后带下划线“_”表示P上方的波浪线【见书82页】
    private INDArray calcEp_xy() {
        INDArray ep_xy = Nd4j.zeros(this.n);   //(x,y)出现的个数

        for (int i = 0; i < this.trainDataArr.columns(); i++) {
            //遍历每对特征(x,y)
            for (Map.Entry<String, Integer> each : this.fxy.get(i).fxy.entrySet()) {
                int index = this.xy2index.get(String.format("%s_%s", i, each.getKey()));  //xy已经进行了拼接

                //计算每种特征的概率[P(x,y)]
                double prob = 1.0 * each.getValue() / this.trainDataArr.rows();
                ep_xy.putScalar(index, prob);
            }
        }

        return ep_xy;
    }

    //计算(x, y)在训练集中出现过的次数
    class IdFxy {
        int id;
        HashMap<String, Integer> fxy;

        public IdFxy(int id) {
            this.id = id;
        }

        public IdFxy(int id, int x, int y) {
            this.id = id;
            this.fxy = new HashMap<>();
            this.fxy.put(String.format("%s_%s", x, y), 1);
        }

        void accumulateFxy(int x, int y) {
            if (fxy == null) {
                this.fxy = new HashMap<>();
            }

            String key = String.format("%s_%s", x, y);

            if (this.fxy.containsKey(key)) {
                this.fxy.put(key, this.fxy.get(key) + 1);
            } else {
                this.fxy.put(key, 1);
            }
        }

        int getFxyCount(int x, int y) {
            return fxy.get(String.format("%s_%s", x, y));
        }

        @Override
        public int hashCode() {
            return super.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            return ((IdFxy) obj).id == this.id;
        }
    }

    //计算(x, y)在训练集中出现过的次数
    private Map<Integer, IdFxy> calcFxy() {
        Map<Integer, IdFxy> result = new HashMap<>();  //按照特征下标来记录(x,y)出现的次数

        for (int i = 0; i < this.trainDataArr.columns(); i++) {
            INDArray column = this.trainDataArr.getColumn(i);
            IdFxy fxy = new IdFxy(i);  //构建一个空的fxy
            for (int j = 0; j < column.length(); j++) {
                int x = this.trainDataArr.getInt(j, i);
                int y = this.trainLabelArr.getInt(j);
                fxy.accumulateFxy(x, y);
            }

            result.put(i, fxy);
        }

        return result;
    }

    private void init() {
        this.fxy = calcFxy();

        //初始训练集中（xi，y）对数量
        this.fxy.forEach((k, v) -> {
            this.n += v.fxy.size();
        });

        //init xy2index,index2xy
        int index = 0;
        for (int i = 0; i < this.trainDataArr.columns(); i++) {
            //遍历每对特征(x,y)
            for (Map.Entry<String, Integer> each : this.fxy.get(i).fxy.entrySet()) {
                String xy = String.format("%s_%s", i, each.getKey());
                this.xy2index.put(xy, index);
                this.index2xy.put(index, xy);
                index++;
            }
        }

        //init ep_xy
        this.ep_xy = calcEp_xy();
    }

    //计算得到的Pw(Y|X)【见书85页】
    private double calcPwy_x(INDArray xi, int y) {
        double num = 0;
        double Z = 0;

        for (int i = 0; i < this.trainDataArr.columns(); i++) {
            String fi_x_y = String.format("%s_%s_%s", i, xi.getInt(i), y);
            if (this.xy2index.containsKey(fi_x_y)) {
                //在xy->index字典中指定当前特征i，以及(x, y)对：(xi, y)，读取其index
                int index = this.xy2index.get(fi_x_y);

                /**
                 * 分子是wi和fi(x，y)的连乘再求和，最后指数
                 * 由于当(x, y)存在时fi(x，y)为1，因为xy对肯定存在，所以直接就是1
                 * 对于分子来说，就是n个wi累加，最后再指数就可以了
                 * 因为有n个w，所以通过id将w与xy绑定，前文的两个搜索字典中的id就是用在这里
                 */
                num += this.w.getInt(index);
            }

            //同时计算其他一种标签y时候的分子，下面的z并不是全部的分母，再加上上式的分子以后才是完整的分母，即z = z + numerator
            fi_x_y = String.format("%s_%s_%s", i, xi.getInt(i), 1 - y);
            if (this.xy2index.containsKey(fi_x_y)) {
                int index = this.xy2index.get(fi_x_y);
                Z += this.w.getInt(index);
            }
        }

        //指数处理
        num = Math.exp(num);
        Z = Math.exp(Z) + num;  //Σ(y1,y2)

        return num / Z;  //正则化处理
    }

    //最大熵模型训练
    private void maxEntropyTrain() {
        this.w = Nd4j.zeros(this.n);

        for (int i = 0; i < this.iterations; i++) {
            System.out.println("start iteration is " + (i + 1));

            //计算“6.2.3 最大熵模型的学习”中的第二个期望（83页最上方那个）
            INDArray epxy = calcEpxy();  //每次迭代都要用到前一步的w来重新计算

            //使用的是IIS，所以设置sigma列表
            INDArray sigmaArr = Nd4j.zeros(this.n);

            for (int j = 0; j < this.n; j++) {
                //依据“6.3.1 改进的迭代尺度法” 式6.34计算
                sigmaArr.putScalar(j, (1.0 / this.M) * Math.log(this.ep_xy.getDouble(j) / epxy.getDouble(j)));
            }

            //按照算法6.1步骤二中的（b）更新w
            this.w = this.w.add(sigmaArr);
        }
    }

    private int predict(INDArray sample) {
        INDArray result = Nd4j.zeros(2);  //创建两个标签的结果

        //每个标签测试下概率
        for (int i = 0; i < result.columns(); i++) {
            result.putScalar(i, this.calcPwy_x(sample, i));
        }

        return BooleanIndexing
                .firstIndex(result, new EqualsCondition(Nd4j.max(result).getDouble(0)))
                .getInt(0);
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
