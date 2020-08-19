package xiuqin.ml.decision_tree;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.conditions.EqualsCondition;
import xiuqin.ml.ModelBase;

public class DecisionTree extends ModelBase {
    static int LABEL_COUNT = 10;
    static int FEATURES = 784;
    int[] featureValues = new int[]{0, 1};  //featureValues

    @Override
    public void normalData(double pivot) {
        //大于pivot为1，小于等于为0
        BooleanIndexing.replaceWhere(this.trainDataArr, 0, Conditions.lessThanOrEqual(pivot));
        BooleanIndexing.replaceWhere(this.trainDataArr, 1, Conditions.greaterThan(pivot));

        BooleanIndexing.replaceWhere(this.testDataArr, 0, Conditions.lessThanOrEqual(pivot));
        BooleanIndexing.replaceWhere(this.testDataArr, 1, Conditions.greaterThan(pivot));
    }

    public static void main(String[] args) {
        DecisionTree dt = new DecisionTree();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        String filePath = "data/Mnist/mnist_train.csv";
        System.out.println("read file:" + filePath);
        dt.loadTrainData(filePath, ",");

        //2、读取测试数据
        filePath = "data/Mnist/mnist_test.csv";
        System.out.println("read file:" + filePath);
        dt.loadTestData(filePath, ",");
        dt.normalData(128);  //正则化处理下数据

        //3、生成DecisionTree
        System.out.println("create decision-tree");
        DTree dTree = dt.createTree(dt.trainDataArr, dt.trainLabelArr);

        //4、进行测试并获得准确率
        System.out.println("testing data");
        double accuracy = dt.modelTest(dTree);
        System.out.println("accuracy rate is " + accuracy);

        //5、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);
    }

    private int getMajorClass(INDArray labelArr) {
        INDArray labelProb = Nd4j.zeros(LABEL_COUNT);

        for (int i = 0; i < LABEL_COUNT; i++) {
            int count = labelArr.scan(new EqualsCondition(i)).intValue();
            labelProb.putScalar(i, count);
        }

        //return max count label
        return BooleanIndexing
                .firstIndex(labelProb, new EqualsCondition(Nd4j.max(labelProb).getInt(0)))
                .getInt(0);
    }

    //empirical entropy
    private double calcEntropy(INDArray labelArr) {
        double HD = 0;

        if (labelArr.isEmpty()) {
            return HD;
        }

        for (int i = 0; i < LABEL_COUNT; i++) {
            int count = labelArr.scan(new EqualsCondition(i)).intValue();
            if (count == 0) continue;
            double prob = 1.0 * count / labelArr.columns();

            HD += -1.0 * prob * (Math.log(prob) / Math.log(2));     //log2(prob)=Math.log(prob) / Math.log(2)
        }

        return HD;
    }

    //conditional entropy
    private double calcCondEntropy(INDArray xiFeature, INDArray sourceLabelArr) {
        double HDA = 0;

        for (int i = 0; i < featureValues.length; i++) {
            //get current feature value
            int value = featureValues[i];

            //count of the value equal one featureValues
            int count = xiFeature.scan(new EqualsCondition(value)).intValue();
            if (count == 0) continue;  //if count==0
            INDArray labelArr = Nd4j.create(count);

            int index = 0;
            for (int j = 0; j < xiFeature.columns(); j++) {
                //if the value equal one featureValues
                if (xiFeature.getInt(j) == value) {
                    labelArr.putScalar(index, sourceLabelArr.getInt(j));   //get the label value at the specified index
                    index++;   //if put one value, then accumulator
                }
            }

            double prob = 1.0 * count / xiFeature.columns();  //P(Di)
            HDA += prob * calcEntropy(labelArr);  //H(D|A)
        }

        return HDA;
    }

    //information gain
    private double[] calcInfoGain(INDArray dataArr, INDArray labelArr) {
        //init max information gain var
        double gain = -1.0;

        //init max feature index
        int index = -1;

        //get empirical entropy
        double entropy = calcEntropy(labelArr);

        //foreach every feature
        for (int i = 0; i < dataArr.columns(); i++) {
            INDArray feature_xi = dataArr.getColumn(i);
            double temp = entropy - calcCondEntropy(feature_xi, labelArr);
            if (temp > gain) {
                gain = temp;
                index = i;
            }
        }

        return new double[]{index, gain};  //return the tuple of index and gain
    }

    private INDArray[] getSubDataArr(INDArray sourceDataArr, INDArray sourceLabelArr, int deleteIndex, int value) {
        int count = sourceDataArr.getColumns(deleteIndex).scan(new EqualsCondition(value)).intValue();
        INDArray dataArr = Nd4j.create(count, sourceDataArr.columns() - 1);   //do not care columns() equal 0 and 1
        INDArray labelArr = Nd4j.create(count);

        if (count > 0) {
            int index = 0;
            for (int i = 0; i < sourceDataArr.rows(); i++) {
                if (sourceDataArr.getInt(i, deleteIndex) == value) {
                    INDArray row = sourceDataArr.getRow(i);
                    INDArray left = row.get(NDArrayIndex.interval(0, deleteIndex));
                    INDArray right = row.get(NDArrayIndex.interval(deleteIndex + 1, row.columns()));
                    row = Nd4j.hstack(left, right);  //dimension is 1
                    dataArr.putRow(index, row);

                    labelArr.putScalar(index, sourceLabelArr.getInt(i));

                    index++;
                }
            }
        }

        return new INDArray[]{dataArr, labelArr};  //return the tuple of sub_data and sub_label
    }

    //ID3 algo
    class DTree {
        int index = -1;   //init index is -1
        int label;
        DTree left, right;

        public DTree(int label) {
            this.label = label;
        }

        public DTree(int index, DTree left, DTree right) {
            this.index = index;
            this.left = left;
            this.right = right;
        }

        public int predict(INDArray dataArr) {
            //if index eaqal -1 ,then the node is leaf
            if (index == -1) {
                return label;
            } else {
                int feature = dataArr.getInt(index);
                if (feature == 0) {
                    return left.predict(dataArr);
                } else {
                    return right.predict(dataArr);
                }
            }
        }
    }

    //use ID3 algo to create DTree
    private DTree createTree(INDArray dataArr, INDArray labelArr) {
        //set epsilon
        double epsilon = 0.1;

        System.out.println(String.format("start a node: feature_count=%d,sample_count=%d", dataArr.columns(), labelArr.columns()));

        //if one feature use dataArr.rows()==1
        if (dataArr.isEmpty() || dataArr.rows() == 1) {
            int label = getMajorClass(labelArr);
            return new DTree(label);
        }

        //if the label are the same
        int label = labelArr.getInt(0);
        int count = labelArr.scan(new EqualsCondition(label)).intValue();  //get the count of values equal labelArr[0]
        if (count == labelArr.columns()) {
            return new DTree(label);
        }

        //get best feature
        double[] ag_epsilon = calcInfoGain(dataArr, labelArr);

        //if < init epsilon
        if (ag_epsilon[1] < epsilon) {
            label = getMajorClass(labelArr);
            return new DTree(label);
        }

        int ag = (int) ag_epsilon[0];

        INDArray[] data_label = getSubDataArr(dataArr, labelArr, ag, 0);   //value is 0
        DTree left = createTree(data_label[0], data_label[1]);

        data_label = getSubDataArr(dataArr, labelArr, ag, 1);  //value is 1
        DTree right = createTree(data_label[0], data_label[1]);

        return new DTree(ag, left, right);
    }

    private int predict(INDArray dataArr, DTree dTree) {
        return dTree.predict(dataArr);
    }

    private double modelTest(DTree dTree) {
        int errorCount = 0;

        int testCount = this.testDataArr.rows();
        for (int i = 0; i < testCount; i++) {
            INDArray each = this.testDataArr.getRow(i);
            long label = predict(each, dTree);   //get predict

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
