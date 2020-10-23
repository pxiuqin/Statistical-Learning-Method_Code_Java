package xiuqin.ml.ext;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

//基于特征值分解实现PCA算法
public class PCA4EVD {
    public static void main(String[] args) {
        PCA4EVD pca=new PCA4EVD();
        INDArray hao = Nd4j.create(new float[]{-1, 1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2}, new int[]{6, 2});
        //System.out.println(hao);

        //PCA h=new PCA(hao);
        //System.out.println(h.reducedBasis(1));

        INDArray result=pca.pca(hao);
        System.out.println(result);
    }

    public INDArray pca(INDArray X) {
        //1、对X中的每列数据做平均
        INDArray mean = X.mean(0);  //以行向量作平均

        //2、正则化处理X元素值
        INDArray norm=X.sub(mean);

        //3、求散度矩阵:X*X^T
        INDArray scatter = norm.transpose().mmul(norm);

        //4、Calculate the eigenvectors and eigenvalues
        assert scatter.rows() == scatter.columns();
        INDArray eigVec = Nd4j.eye(scatter.rows());  //生成单位矩阵
        INDArray eigVal = Eigen.symmetricGeneralizedEigenvalues(eigVec, scatter, true);

        //5、sort eig_vec based on eig_val from highest to lowest
        int maxIndex=Nd4j.argMax(eigVal).getInt(0);

        //6、select the top k eig_vec
        INDArray feature = eigVec.getRow(maxIndex);  //降成1维

        //7、生成一个新data
        return X.mmul(feature.reshape(X.columns()));
    }

    /*public static INDArray[] covarianceMatrix(INDArray in) {
        long dlength = in.rows();
        long vlength = in.columns();

        INDArray sum = Nd4j.create(vlength);
        INDArray product = Nd4j.create(vlength, vlength);

        for (int i = 0; i < vlength; i++)
            sum.getColumn(i).assign(in.getColumn(i).sumNumber().doubleValue() / dlength);

        for (int i = 0; i < dlength; i++) {
            INDArray dx1 = in.getRow(i).sub(sum);
            product.addi(dx1.reshape(vlength, 1).mmul(dx1.reshape(1, vlength)));
        }
        product.divi(dlength);
        return new INDArray[]{product, sum};
    }*/

    public static INDArray[] covarianceMatrix(INDArray in) {
        long dlength = (long)in.rows();
        long vlength = (long)in.columns();
        INDArray product = Nd4j.create(new long[]{vlength, vlength});
        INDArray sum = in.sum(new int[]{0}).divi(dlength);   //变成均值

        for(int i = 0; (long)i < dlength; ++i) {
            INDArray dx1 = in.getRow((long)i).sub(sum);
            product.addi(dx1.reshape(vlength, 1L).mmul(dx1.reshape(1L, vlength)));
        }

        product.divi(dlength);
        return new INDArray[]{product, sum};
    }
}
