package xiuqin.ml.ext;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SVD {
    public static void main(String[] args) {
        INDArray A = Nd4j.create(new float[]{0, 1, 1, 1, 1, 0}, new int[]{3, 2});
        System.out.println(A);

        long m = A.rows();
        long n = A.columns();

        // The prepare SVD results, we'll decomp A to UxSxV'
        INDArray S = Nd4j.create(A.dataType(), m < n ? m : n);

        INDArray VT = Nd4j.create(A.dataType(), new long[]{n, n}, 'f');

        INDArray U = Nd4j.create(A.dataType(), new long[]{m, m}, 'f');

        // Note - we don't care about U
        //Nd4j.getBlasWrapper().lapack().gesvd(A, S, U, VT);
        gesvd(A, S, U, VT);

        System.out.println(U);
        System.out.println(S);
        System.out.println(VT);
    }

    //calc UxSxV'
    public static void gesvd(INDArray A, INDArray S, INDArray U, INDArray VT) {
        //calc V=A^T*A
        INDArray ATA = A.transpose().mmul(A);

        //calc ATA eig_vec and eig_val
        INDArray eigVec_ATA = Nd4j.eye(ATA.rows());  //生成单位矩阵
        Eigen.symmetricGeneralizedEigenvalues(eigVec_ATA, ATA, true);

        //calc U=A*A^T
        INDArray AAT = A.mmul(A.transpose());

        System.out.println(ATA);
        System.out.println(AAT);
        //calc AAT eig_vec and eig_val
        INDArray eigVec_AAT = Nd4j.eye(AAT.rows());  //生成单位矩阵
        INDArray eigVal_AAT = Eigen.symmetricGeneralizedEigenvalues(eigVec_AAT, AAT);   //Eigen is error, AAT is singular matrix

        //calc S=AV/U
        //INDArray Avi = A.mmul(eigVec_ATA).div(eigVec_AAT);
        /*for (int i = 0; i < eigVec_ATA.rows(); i++) {
            INDArray vi = eigVec_ATA.getRow(i);
            INDArray si = A.mmul(vi).div(eigVec_AAT.getRow(i));   //A*vi=si*ui

        }*/

        //calc S=eig_val^1/2
        S = Transforms.sqrt(eigVal_AAT).dup();
        //S = InvertMatrix.invert(Nd4j.diag(S), false);

        U = eigVec_AAT.dup();
        VT = eigVec_ATA.dup();
    }
}
