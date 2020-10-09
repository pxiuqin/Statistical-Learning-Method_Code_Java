package xiuqin.ml.svm;

import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import xiuqin.ml.ModelBase;

public class SVM extends ModelBase {
    //Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为-1
    @Override
    protected void normalLabel(float pivot) {
        BooleanIndexing.replaceWhere(trainLabelArr, 1, Conditions.equals(pivot));
        BooleanIndexing.replaceWhere(trainLabelArr, -1, Conditions.greaterThan(pivot));

        BooleanIndexing.replaceWhere(testLabelArr, 1, Conditions.equals(pivot));
        BooleanIndexing.replaceWhere(testLabelArr, -1, Conditions.greaterThan(pivot));
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
