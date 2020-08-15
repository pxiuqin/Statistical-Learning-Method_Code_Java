package xiuqin.ml;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import xiuqin.utils.FileUtils;

public abstract class ModelBase {
    protected INDArray trainLabelArr;
    protected INDArray trainDataArr;

    protected INDArray testLabelArr;
    protected INDArray testDataArr;


    //Mnsit data have 0-9 lable, two class for 1 of >=5, -1 of <5
    protected void normalLabel(float pivot) {
        BooleanIndexing.replaceWhere(trainLabelArr, -1, Conditions.lessThan(pivot));
        BooleanIndexing.replaceWhere(trainLabelArr, 1, Conditions.greaterThanOrEqual(pivot));

        /*
        for (int i = 0; i < trainLabelArr.columns(); i++) {
            trainLabelArr.putScalar(i, trainLabelArr.getFloat(i) >= point ? 1 : -1);
        }*/

        BooleanIndexing.replaceWhere(testLabelArr, -1, Conditions.lessThan(pivot));
        BooleanIndexing.replaceWhere(testLabelArr, 1, Conditions.greaterThanOrEqual(pivot));

        /*
        for (int i = 0; i < testLabelArr.columns(); i++) {
            testLabelArr.putScalar(i, testLabelArr.getFloat(i) >= point ? 1 : -1);
        }*/
    }

    //normal
    protected void normalData(double sum) {
        trainDataArr = trainDataArr.div(sum);
        testDataArr = testDataArr.div(sum);
    }

    //load raw data
    private int[] genDataColumn(int cols) {
        int[] result = new int[(cols - 1)];
        for (int i = 0; i < cols - 1; i++) {
            result[i] = i + 1;
        }

        return result;
    }

    protected int[] genColumnSeq(int cols) {
        int[] result = new int[(cols)];
        for (int i = 0; i < cols; i++) {
            result[i] = i;
        }

        return result;
    }

    //load train data
    protected void loadTrainData(String path, String separator) {
        try {
            INDArray result = FileUtils.readFromText(path, separator);

            //拆分第一列数据为label
            this.trainLabelArr = result.getColumn(0);

            //剩下的数据为data
            this.trainDataArr = result.getColumns(genDataColumn(result.columns()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //load test data
    protected void loadTestData(String path, String separator) {
        try {
            INDArray result = FileUtils.readFromText(path, separator);

            //拆分第一列数据为label
            this.testLabelArr = result.getColumn(0);

            //剩下的数据为data
            //this.testDataArr = result.getColumns(genDataColumn(result.columns()));
            this.testDataArr = result.get(NDArrayIndex.all(), NDArrayIndex.interval(1, result.columns()));  //all row, n-1 columns
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}