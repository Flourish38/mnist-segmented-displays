using MLDatasets, ImageShow, ImageIO, FileIO, Statistics

data = MNIST(:train)
(X, y) = data.features, data.targets
convert2image(MNIST, X[:, :, 1])
encoding = BitMatrix([
    0 1 1 1 0 1 1 1;
    0 0 0 1 0 0 1 0;
    0 1 0 1 1 1 0 1;
    0 1 0 1 1 0 1 1;
    0 0 1 1 1 0 1 0;
    0 1 1 0 1 0 1 1;
    0 1 1 0 1 1 1 1;
    0 1 0 1 0 0 1 0;
    0 1 1 1 1 1 1 1;
    0 1 1 1 1 0 1 1
])'
indices = [y .== n for n in 0:9]

pixel = CartesianIndex(14, 14)
pixel_data = [mean(X[pixel, i]) for i in indices]
errors = [mse(pixel_data, segment_data) for segment_data in eachrow(encoding)]

mse(x, y) = mean((x .- y) .^ 2)

output = similar(X[:, :, 1], Int)
for pixel in CartesianIndices((1:28, 1:28))
    pixel_data = [mean(X[pixel, i]) for i in indices]
    errors = [mse(pixel_data, segment_data) for segment_data in eachrow(encoding)]
    output[pixel] = argmin(errors)
end

for (n, segments) in enumerate(eachcol(encoding))
    image = falses(size(output))
    for (i, segment) in enumerate(segments)
        image[output.==i] .|= segment
    end
    save("output_images/straightforward_$(n-1).png", simshow(image))
end