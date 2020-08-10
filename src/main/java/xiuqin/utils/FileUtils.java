package xiuqin.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class FileUtils {
    public static List<String> readString(File file) throws IOException {
        List<String> pairs = new ArrayList<String>();
        InputStreamReader read = new InputStreamReader(new FileInputStream(file));
        BufferedReader bufferedReader = new BufferedReader(read);
        String lineTxt = null;
        while ((lineTxt = bufferedReader.readLine()) != null) {
            pairs.add(lineTxt);
        }
        read.close();

        return pairs;
    }

    public static INDArray readMatrix(String filePaht, String separator) throws Exception {
        File file = new File(filePaht);
        List<INDArray> result = new ArrayList<>();
        InputStreamReader read = new InputStreamReader(new FileInputStream(file));
        BufferedReader bufferedReader = new BufferedReader(read);
        String lineTxt = null;
        while ((lineTxt = bufferedReader.readLine()) != null) {
            String[] col = lineTxt.trim().split(separator);

            //数据类型转换并生成一个INDArray
            result.add(Nd4j.create(Arrays.asList(col).stream().map(e -> Long.parseLong(e)).collect(Collectors.toList())));
        }

        return Nd4j.vstack(result);
    }

    public static INDArray readFromText(String filePaht, String separator) throws IOException {
        return Nd4j.readNumpy(filePaht, separator);
    }
}
