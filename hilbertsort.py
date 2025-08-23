import numpy as np

class HilbertSort3D(object):

    def __init__(self, origin=(0,0,0), radius=1.0, bins=32):
        '''
        '''
        self.origin = np.array(origin)
        self.radius = radius
        self.bins = bins
        order = np.log2(32)
        if order%1.0 > 0.0: raise ValueError("HilbertSort: Bins should be a power of 2.")
        self.curve = self._hilbert_3d(int(np.log2(bins)))

    def _hilbert_3d(self, order):
            '''
            Method generates 3D hilbert curve of desired order.
            Param:
                order - int ; order of curve
            Returns:
                np.array ; list of (x, y, z) coordinates of curve
            '''

            def gen_3d(order, x, y, z, xi, xj, xk, yi, yj, yk, zi, zj, zk, array):
                if order == 0:
                    xx = x + (xi + yi + zi)/3
                    yy = y + (xj + yj + zj)/3
                    zz = z + (xk + yk + zk)/3
                    array.append((xx, yy, zz))
                else:
                    gen_3d(order-1, x, y, z, yi/2, yj/2, yk/2, zi/2, zj/2, zk/2, xi/2, xj/2, xk/2, array)

                    gen_3d(order-1, x + xi/2, y + xj/2, z + xk/2,  zi/2, zj/2, zk/2, xi/2, xj/2, xk/2,
                               yi/2, yj/2, yk/2, array)
                    gen_3d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, z + xk/2 + yk/2, zi/2, zj/2, zk/2,
                               xi/2, xj/2, xk/2, yi/2, yj/2, yk/2, array)
                    gen_3d(order-1, x + xi/2 + yi, y + xj/2+ yj, z + xk/2 + yk, -xi/2, -xj/2, -xk/2, -yi/2,
                               -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                    gen_3d(order-1, x + xi/2 + yi + zi/2, y + xj/2 + yj + zj/2, z + xk/2 + yk +zk/2, -xi/2,
                               -xj/2, -xk/2, -yi/2, -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                    gen_3d(order-1, x + xi/2 + yi + zi, y + xj/2 + yj + zj, z + xk/2 + yk + zk, -zi/2, -zj/2,
                               -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                    gen_3d(order-1, x + xi/2 + yi/2 + zi, y + xj/2 + yj/2 + zj , z + xk/2 + yk/2 + zk, -zi/2,
                               -zj/2, -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                    gen_3d(order-1, x + xi/2 + zi, y + xj/2 + zj, z + xk/2 + zk, yi/2, yj/2, yk/2, -zi/2, -zj/2,
                               -zk/2, -xi/2, -xj/2, -xk/2, array)

            n = pow(2, order)
            hilbert_curve = []
            gen_3d(order, 0, 0, 0, n, 0, 0, 0, n, 0, 0, 0, n, hilbert_curve)

            return np.array(hilbert_curve).astype('int')

    def sort(self, data):
        '''
        Method bins points according to parameters and sorts by traversing binning
        matrix using hilbert space-filling curve.
        Param:
            data - np.array; list of 3D points; (Nx3)
        Returns:
            sorted_data - np.array; list of sorted 3D points; (Nx3)

        '''
        # Center data around origin
        data_ = data - self.origin

        # Bin points
        binned = [[[[] for k in range(self.bins)] for j in range(self.bins)] for i in range(self.bins)]
        bin_interval = ((self.radius*2) / self.bins)
        offset = int(self.radius/bin_interval)
        for i, _ in enumerate(data):
            x = int(_[-3]/bin_interval) + offset
            y = int(_[-2]/bin_interval) + offset
            z = int(_[-1]/bin_interval) + offset
            if (x > self.bins-1) or (x < 0): continue
            if (y > self.bins-1) or (y < 0): continue
            if (z > self.bins-1) or (z < 0): continue
            binned[x][y][z].append(_)

        # Traverse and Assemble
        sorted_data = []
        for _ in self.curve:
            x = binned[_[0]][_[1]][_[2]]
            if len(x) > 0: sorted_data.append(np.array(x))
        sorted_data = np.concatenate(sorted_data, axis=0)

        return sorted_data