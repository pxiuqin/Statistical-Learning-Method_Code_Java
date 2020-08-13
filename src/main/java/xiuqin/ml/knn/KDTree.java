package xiuqin.ml.knn;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.Vector;

/*
 * KDTree implementation
 * Features :
 * K-Dimension
 * Search : Range Search, Nearest NeighBor Search
 * Insert : SingleNode insert, Points set insert (Split by median using median of medians and presort)
 * Refer to https://en.wikipedia.org/wiki/K-d_tree
 * Author : linpc2013:https://github.com/linpc2013/KDTree/blob/master/KDTree/src/code/KDTree.java
 *
 * */
// K - Dimension Point
class HyperPoint {
    double[] coords;
    int K = 0;

    public HyperPoint(double[] crds) {
        if (crds == null)
            throw new NullPointerException("");
        K = crds.length;
        coords = new double[K];
        for (int i = 0; i < K; i++)
            coords[i] = crds[i];
    }

    public HyperPoint(HyperPoint p) {
        this(p.coords);
    }

    public boolean equals(HyperPoint p) {
        if (K != p.K)
            throw new IllegalArgumentException("");
        for (int i = 0; i < K; i++)
            if (p.coords[i] != coords[i])
                return false;
        return true;
    }

    // Euclidean Distance
    public double distanceTo(HyperPoint p) {
        return Math.sqrt(squareDistanceTo(p));
    }

    public double squareDistanceTo(HyperPoint p) {
        if (K != p.K)
            throw new IllegalArgumentException("");
        double res = 0;
        for (int i = 0; i < K; i++)
            res += (coords[i] - p.coords[i]) * (coords[i] - p.coords[i]);
        return res;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < K; i++)
            sb.append(coords[i] + " ");
        return sb.toString();
    }
}

// K - Dimension Space

class HyperSpace {
    HyperPoint min, max;
    int K = 0;

    public HyperSpace(HyperPoint min, HyperPoint max) {
        if (min == null || max == null)
            throw new NullPointerException("");
        K = min.K;
        if (K == 0 || K != max.K)
            throw new IllegalArgumentException("");
        this.min = new HyperPoint(min);
        this.max = new HyperPoint(max);
    }

    // Detect whether intersects with other HyperSpace or not
    public boolean intersects(HyperSpace p) {
        for (int i = 0; i < K; i++)
            if (min.coords[i] > p.max.coords[i] || max.coords[i] < p.min.coords[i])
                return false;
        return true;
    }

    public boolean contains(HyperPoint p) {
        if (K != p.K)
            throw new IllegalArgumentException("");
        for (int i = 0; i < K; i++)
            if (min.coords[i] > p.coords[i] || p.coords[i] > max.coords[i])
                return false;
        return true;
    }

    // The square of Euclidean Distance
    public double squareDistanceTo(HyperPoint p) {
        if (K != p.K)
            throw new IllegalArgumentException("");
        double res = 0;
        for (int i = 0; i < K; i++)
            if (min.coords[i] > p.coords[i])
                res += (min.coords[i] - p.coords[i]) * (min.coords[i] - p.coords[i]);
            else if (p.coords[i] > max.coords[i])
                res += (p.coords[i] - max.coords[i]) * (p.coords[i] - max.coords[i]);
        return res;
    }

    // Euclidean Distance
    public double distanceTo(HyperPoint p) {
        return Math.sqrt(squareDistanceTo(p));
    }

    public String toString() {
        return min.toString() + "->" + max.toString();
    }
}

public class KDTree {
    class Node {
        // HyperSpace hs is used to accelerate range search
        HyperSpace hs;

        // Current spliting node
        HyperPoint p;
        Node left, right;

        public Node(HyperSpace hs, HyperPoint p) {
            this.hs = hs;
            this.p = p;
            left = right = null;
        }
    }

    Node root;
    int K = 2;  //默认维度
    double RANGE = 1.0;

    // HyperPoint min, max are determined the range of KDTree Space
    HyperPoint min, max;

    public KDTree(int K) {
        this.K = K;
        root = null;
        double[] vals = new double[K];
        min = new HyperPoint(vals);
        for (int i = 0; i < K; i++)
            vals[i] = RANGE;
        max = new HyperPoint(vals);
    }

    public KDTree(int K, HyperPoint min, HyperPoint max) {
        this.K = K;
        this.min = min;
        this.max = max;
        root = null;
    }

    /*
     * Single Node insertion just like binary search tree but be careful to the
     * cycle of coordinate
     */
    public void insert(HyperPoint p) {
        HyperPoint hmin = new HyperPoint(min);
        HyperPoint hmax = new HyperPoint(max);

        //首次插入根节点为null，当前p为根结点，后续每次把p进行k维切分后，保存在root中
        root = insert(root, p, hmin, hmax, 0);
    }

    //递归插入，这里不考虑切分使用中位数开始
    private Node insert(Node r, HyperPoint p, HyperPoint hmin, HyperPoint hmax, int depth) {
        //if root is null
        if (r == null)
            //根结点构造或叶子节点，根结点对应k维空间中包含所有实例点的超矩形区域，叶子节点的话保存当前实例点
            return new Node(new HyperSpace(hmin, hmax), p);

        //通过递归方法，不断对k维空间进行切分，生成子结点。在超矩形区域（结点）上选择一个坐标轴和在此坐标轴上的一个切分点
        int k = depth % K;  //对深度为depth的结点，选择切分的坐标轴为：depth[mod(K)]+1,这里没有+1？
        double pivot = r.p.coords[k];  //选择切分点

        //通过切分点确定一个超平面，这个超平面通过选定的切分点并垂直于选定的坐标轴，将当前超矩形区域切分为左右两个子区域（子结点）
        if (p.coords[k] < pivot) {  //left node：左子结点对应坐标小于切分点的子区域
            hmax.coords[k] = pivot;  //next max node is pivot

            //the left node of current node：将落在切分超平面上的实例点保存在根节点，
            //其实当前left作为下一步的根，如果有left继续划分，没有就把当前节点保存成left
            r.left = insert(r.left, p, hmin, hmax, depth + 1);
        } else {
            hmin.coords[k] = pivot;  //下一切分区域的最小边界就是当前切分点
            r.right = insert(r.right, p, hmin, hmax, depth + 1);
        }

        return r;
    }

    // Presort method
    // Inner class SortComparator is used for presort of points set
    class SortComparator implements Comparator<HyperPoint> {
        int k;

        public void setK(int k) {
            this.k = k;
        }

        @Override
        public int compare(HyperPoint o1, HyperPoint o2) {
            if (o1.coords[k] > o2.coords[k])
                return 1;
            else if (o1.coords[k] == o2.coords[k])
                return 0;
            return -1;
        }
    }

    public void insertByPreSort(HyperPoint[] points) {
        int num = points.length;
        HyperPoint hmin = new HyperPoint(min);
        HyperPoint hmax = new HyperPoint(max);

        // k presort points set
        HyperPoint[][] kpoints = new HyperPoint[K][];
        SortComparator sc = new SortComparator();

        // Presort【K个维度分别排序】
        for (int k = 0; k < K; k++) {
            sc.setK(k);
            Arrays.sort(points, sc);
            kpoints[k] = points.clone();
        }

        Vector<HyperPoint> avails = new Vector<HyperPoint>();
        for (int i = 0; i < num; i++)
            avails.add(kpoints[0][i]);  //用了x维度排序

        root = insertByPreSort(root, kpoints, hmin, hmax, 0, avails);
    }

    //排序后进行KDTree构建
    private Node insertByPreSort(Node r, HyperPoint[][] kpoints, HyperPoint hmin, HyperPoint hmax, int depth, Vector<HyperPoint> avails) {
        int num = avails.size();  //插入节点数

        if (num == 0)
            return null;
        else {
            int k = depth % K;
            if (num == 1)
                return new Node(new HyperSpace(hmin, hmax), avails.get(0));  //只插入一个节点
            int mid = (num - 1) / 2;  //找到中位数下标
            if (r == null)
                r = new Node(new HyperSpace(hmin, hmax), avails.get(mid));  //构建根结点或叶子结点
            HyperPoint hmid1 = new HyperPoint(hmax);
            hmid1.coords[k] = kpoints[k][mid].coords[k];  //确定切分点

            // Splitting current points set
            HashMap<HyperPoint, Integer> split = new HashMap<HyperPoint, Integer>();
            for (int p = 0; p < num; p++)
                if (p < mid)
                    split.put(avails.get(p), 0);
                else if (p > mid)
                    split.put(avails.get(p), 1);

            int k1 = (depth + 1) % K;  //nest depth

            // Generating left and right branch available points set
            Vector<HyperPoint> left = new Vector<HyperPoint>(), right = new Vector<HyperPoint>();
            for (HyperPoint p : kpoints[k1])  //确定那个排序维度
                if (split.containsKey(p))
                    if (split.get(p) == 0)
                        left.addElement(p);
                    else
                        right.addElement(p);

            // Recursive Split
            r.left = insertByPreSort(r.left, kpoints, hmin, hmid1, depth + 1, left);  //左
            HyperPoint hmid2 = new HyperPoint(hmin);
            hmid1.coords[k] = kpoints[k][mid].coords[k];
            r.right = insertByPreSort(r.right, kpoints, hmid2, hmax, depth + 1, right);

            return r;
        }
    }

    public void insertByMedianFinding(HyperPoint[] points) {
        int num = points.length;
        HyperPoint hmin = new HyperPoint(min);
        HyperPoint hmax = new HyperPoint(max);
        root = insertByMedianFinding(root, points, hmin, hmax, 0, 0, num - 1);
    }

    // quickSort partition function
    private int partition(HyperPoint[] points, int k, int beg, int end) {
        HyperPoint pivot = points[beg];
        int i = beg, j = end + 1;
        while (true) {
            while (++i <= end && points[i].coords[k] < pivot.coords[k])
                ;  //从小于基准值开始
            while (--j > beg && points[j].coords[k] >= pivot.coords[k])
                ;  //从大于等于基准值开始
            if (i < j) {  //交换
                HyperPoint temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            } else
                break;
        }
        points[beg] = points[j];
        points[j] = pivot;
        return j;
    }

    // median of medians algorithm【目的是找中位数，这里通过参数k来确定是使用那个维度】
    // Refer to https://en.wikipedia.org/wiki/Median_of_medians
    private int findMedian(HyperPoint[] points, int k, int beg, int end) {
        if (beg > end)
            return -1;
        else if (beg == end)
            return beg;
        int mid = (beg + end) / 2;
        int i = beg, j = end;
        while (true) {
            int t = partition(points, k, i, j);  //快排，看是否中位数位置已经排序了，如果排序了就停止
            if (t == mid)
                return t;
            else if (t > mid)
                j = t - 1;   //去掉高于t的部分
            else
                i = t + 1;  //去掉低于t的部分
        }
    }

    private Node insertByMedianFinding(Node r, HyperPoint[] points, HyperPoint hmin, HyperPoint hmax, int depth, int i, int j) {
        if (i > j)
            return null;
        else if (i == j)
            return new Node(new HyperSpace(hmin, hmax), points[i]);
        int k = depth % K;

        // Find the index of median
        int t = findMedian(points, k, i, j);  //中位数下标
        HyperPoint p = points[t];
        if (r == null)
            r = new Node(new HyperSpace(hmin, hmax), p);

        double pivot = p.coords[k];  //切分点
        HyperPoint hmid1 = new HyperPoint(hmax);
        hmid1.coords[k] = p.coords[k];
        r.left = insertByMedianFinding(r.left, points, hmin, hmid1, depth + 1, i, t - 1);
        HyperPoint hmid2 = new HyperPoint(hmin);
        hmid2.coords[k] = pivot;
        r.right = insertByMedianFinding(r.right, points, hmid2, hmax, depth + 1, t + 1, j);

        return r;
    }

    /*
     * Nearest Neighbor Finding Record the the node of current best, and
     * continue check the the distance between current node and input node, if
     * distance is smaller, then update current best. Using pruning strategy to
     * prune left or right branch of current node. If input node is smaller than
     * current node r in the x coordinate, then algorithm will check left
     * branch. But only if the hypersphere whose center is input node and radius
     * is current minimal distance intersects with right branch, algorithm check
     * the right branch.
     */
    // current best node
    HyperPoint nmin;
    // current minimal distance
    double ndist;

    public HyperPoint nearestPoint(HyperPoint p) {
        if (root == null)
            return null;
        nmin = root.p;
        ndist = nmin.squareDistanceTo(p);
        nearestPoint(root, p, 0);
        return nmin;
    }

    private void nearestPoint(Node r, HyperPoint p, int depth) {
        if (r == null)
            return;
        double dist = r.p.squareDistanceTo(p);

        // update current best
        if (dist < ndist) {
            nmin = r.p;
            ndist = dist;
        }
        int k = depth % K;
        double pivot = r.p.coords[k];  //切分点
        if (p.coords[k] < pivot) {
            nearestPoint(r.left, p, depth + 1);
            // Hyper space intersect with right branch
            if (p.coords[k] + Math.sqrt(ndist) >= pivot)
                nearestPoint(r.right, p, depth + 1);
        } else {
            nearestPoint(r.right, p, depth + 1);
            if (p.coords[k] - Math.sqrt(ndist) <= pivot)
                nearestPoint(r.left, p, depth + 1);
        }
    }

    /*
     * Range Search A simple implementation using recursion if current node's
     * hyperSpace doesn't intersect with required range, then current node will
     * be ignore. Otherwise, check the left or right son of current node.
     */
    public Set<HyperPoint> rangeQuery(HyperSpace hs) {
        Set<HyperPoint> res = new HashSet<HyperPoint>();
        rangeQuery(root, hs, res);
        return res;
    }

    private void rangeQuery(Node r, HyperSpace hs, Set<HyperPoint> res) {
        // If current node r is null or doesn't intersect with hs, then return
        if (r == null || !r.hs.intersects(hs))
            return;
        if (hs.contains(r.p))
            res.add(r.p);
        // recursively check the left, right branch of current node
        rangeQuery(r.left, hs, res);
        rangeQuery(r.right, hs, res);
    }

    // Test Code
    public static void main(String[] args) {
        double[][] ps = {{0.2, 0.3, 0.4}, {0.3, 0.4, 0.5}, {0.1, 0.7, 0.3}, {0.1, 0.2, 0.9}};
        int num = ps.length;
        HyperPoint[] hps = new HyperPoint[num];
        for (int i = 0; i < num; i++)
            hps[i] = new HyperPoint(ps[i]);
        double[][] range = {{0, 0, 0}, {1, 1, 1}};
        HyperPoint min = new HyperPoint(range[0]);
        HyperPoint max = new HyperPoint(range[1]);
        int K = range[0].length;
        KDTree kd = new KDTree(K, min, max);
        // Insert
        // ---------------------------------------
        // I. Single Point Insert
        // for (int i = 0; i < num; i++)
        // kd.insert(hps[i]);

        // II. Insert Points set by O(n) Median Find Algorithm
        kd.insertByMedianFinding(hps);

        // III. Using PreSort to fast insert Point Set
        // kd.insertByPreSort(hps);
        double[] ps4 = {1, 1, 1};
        HyperPoint hp4 = new HyperPoint(ps4);

        // Nearest Point search
        // ---------------------------------------
        // hp5 should be { 0.3, 0.4, 0.5 }
        HyperPoint hp5 = kd.nearestPoint(hp4);

        // Range search
        // ---------------------------------------
        // qu should contain { { 0.2, 0.3, 0.4 }, { 0.3, 0.4, 0.5 }}
        double[][] range1 = {{0, 0, 0}, {0.5, 0.5, 0.5}};
        Set<HyperPoint> qu = kd.rangeQuery(new HyperSpace(new HyperPoint(range1[0]), new HyperPoint(range1[1])));
    }
}