function vectorIndex(nx,i,j)
    return (i-1)*nx + j;
end

function Vec2Mat(nx,ny,v)
    m = zeros(nx,ny);
    for i = 1:nx
        for j = 1:ny
            m[i,j] = v[(i-1)*ny + j]
        end
    end
    return m;
end

function Mat2Vec(m)
    nx = size(m,1);
    ny = size(m,2);
    v = zeros(nx*ny);
    for i = 1:nx
        for j = 1:ny
            v[(i-1)*ny + j] = m[i,j];
        end
    end
    return v;
end