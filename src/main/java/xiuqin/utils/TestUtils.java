package xiuqin.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;

public class TestUtils {
    public static void main(String[] args) {
        //测试Nd4j中的add和addi方法的区别
        testAddi();
    }

    private static void testAddi() {
        INDArray a = Nd4j.ones(2, 2);
        System.out.println(a);
        a.add(1);
        System.out.println(a);
        a.addi(1);
        System.out.println(a);
    }
}
