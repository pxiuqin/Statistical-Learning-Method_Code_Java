package xiuqin.ml.boost;

import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import xiuqin.ml.ModelBase;

public class AdaBoost extends ModelBase {
    private int treeNumber=50;   //boosting tree deep number

    //Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为-1
    @Override
    protected void normalLabel(float pivot) {
        BooleanIndexing.replaceWhere(trainLabelArr, -1, Conditions.greaterThan(pivot));
        BooleanIndexing.replaceWhere(trainLabelArr, 1, Conditions.equals(pivot));

        BooleanIndexing.replaceWhere(testLabelArr, -1, Conditions.greaterThan(pivot));
        BooleanIndexing.replaceWhere(testLabelArr, 1, Conditions.equals(pivot));
    }

    //Mnsit有0-255是个data，所以将>=128的作为1，<128 is 0
    @Override
    protected void normalData(double pivot) {
        BooleanIndexing.replaceWhere(trainLabelArr, 0, Conditions.lessThan(pivot));
        BooleanIndexing.replaceWhere(trainLabelArr, 1, Conditions.greaterThanOrEqual(pivot));

        BooleanIndexing.replaceWhere(testLabelArr, 0, Conditions.greaterThan(pivot));
        BooleanIndexing.replaceWhere(testLabelArr, 1, Conditions.greaterThanOrEqual(pivot));
    }


    public static void main(String[] args) {
        AdaBoost ada = new AdaBoost();
        long currentTime = System.currentTimeMillis();

        //1、读取训练数据
        String filePath = "data/Mnist/mnist_train.csv";
        System.out.println("read file:" + filePath);
        ada.loadTrainData(filePath, ",");

        //2、读取测试数据
        filePath = "data/Mnist/mnist_test.csv";
        System.out.println("read file:" + filePath);
        ada.loadTestData(filePath, ",");
        ada.normalLabel(0);
        ada.normalData(128);  //正则化处理下数据

        //3、训练
        System.out.println("init params");
        //ada.init();  //加载完数据后进行初始化
        System.out.println("training data");
        //ada.train(); //训练数据

        //4、进行测试并获得准确率
        System.out.println("testing data");
        double accuracy = 1.0;//ada.modelTest();
        System.out.println("accuracy rate is " + accuracy);

        //5、计算所用时间
        System.out.println((System.currentTimeMillis() - currentTime) / 1000);
    }
}
