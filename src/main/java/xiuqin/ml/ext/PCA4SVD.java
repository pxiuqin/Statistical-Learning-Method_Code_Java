package xiuqin.ml.ext;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;

//use svd for pca
public class PCA4SVD {
    public static void main(String[] args) {
        PCA4SVD pca = new PCA4SVD();
        INDArray hao = Nd4j.create(new float[]{-1, 1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2}, new int[]{6, 2});
        System.out.println(hao);

        //PCA h = new PCA(hao);
        //System.out.println(h.estimateVariance(hao.getRow(5),1));

        INDArray result = pca.pca(hao, 1, true);
        System.out.println(result);
    }

    public INDArray pca(INDArray A, int nDims, boolean normalize) {
        INDArray factor = pca_factor(A, nDims, normalize);
        return A.mmul(factor);
    }

    //use svd directly without eigenvalue decomposition
    private INDArray pca_factor(INDArray A, int nDims, boolean normalize) {

        if (normalize) {
            // Normalize to mean 0 for each feature ( each column has 0 mean )
            INDArray mean = A.mean(0);
            A.subiRowVector(mean);
        }

        long m = A.rows();
        long n = A.columns();

        // The prepare SVD results, we'll decomp A to UxSxV'
        INDArray s = Nd4j.create(A.dataType(), m < n ? m : n);

        INDArray VT = Nd4j.create(A.dataType(), new long[]{n, n}, 'f');

        // Note - we don't care about U
        Nd4j.getBlasWrapper().lapack().gesvd(A, s, null, VT);

        // for comparison k & nDims are the equivalent values in both methods implementing PCA

        // So now let's rip out the appropriate number of left singular vectors from
        // the V output (note we pulls rows since VT is a transpose of V)
        INDArray V = VT.transpose();  //use V
        INDArray factor = Nd4j.create(A.dataType(), new long[]{n, nDims}, 'f');
        for (int i = 0; i < nDims; i++) {
            factor.putColumn(i, V.getColumn(i));
        }

        return factor;
    }
}
