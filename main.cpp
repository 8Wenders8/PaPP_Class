#include <iostream>
#include <string>
#include <ctime>
#include <png.h>
#include <mpi.h>
#include <chrono>

typedef struct png_data {
   unsigned int width, height;
   png_byte color_type, bit_depth;
   png_bytep *row_pointers;
}PNG_DATA;

void read_png_file(char *filename, PNG_DATA* data) {
    FILE *fp = fopen(filename, "rb");
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) abort();
    png_infop info = png_create_info_struct(png);
    if(!info) abort();
    if(setjmp(png_jmpbuf(png))) abort();
    png_init_io(png, fp);
    png_read_info(png, info);
    data->width = png_get_image_width(png, info);
    data->height = png_get_image_height(png, info);
    data->color_type = png_get_color_type(png, info);
    data->bit_depth = png_get_bit_depth(png, info);

    if(data->bit_depth == 16)
        png_set_strip_16(png);
    if(data->color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if(data->color_type == PNG_COLOR_TYPE_GRAY && data->bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if(data->color_type == PNG_COLOR_TYPE_RGB ||
       data->color_type == PNG_COLOR_TYPE_GRAY ||
       data->color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(data->color_type == PNG_COLOR_TYPE_GRAY ||
       data->color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    png_read_update_info(png, info);
    if (data->row_pointers) abort();
    data->row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * data->height);
    for(int y = 0; y < data->height; y++) {
        data->row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }
    png_read_image(png, data->row_pointers);
    fclose(fp);
    png_destroy_read_struct(&png, &info, nullptr);
}

void write_png_file(char *filename, PNG_DATA *data) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) abort();
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();
    png_infop info = png_create_info_struct(png);
    if (!info) abort();
    if (setjmp(png_jmpbuf(png))) abort();
    png_init_io(png, fp);
    // Output is 8bit depth, RGBA format.
    png_set_IHDR(png, info, data->width, data->height,8,
            PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT );
    png_write_info(png, info);

    if (!data->row_pointers) abort();
    png_write_image(png, data->row_pointers);
    png_write_end(png, nullptr);

    for(int y = 0; y < data->height; y++) {
        free(data->row_pointers[y]);
    }
    free(data->row_pointers);
    fclose(fp);
    png_destroy_write_struct(&png, &info);
}


void process_png_file(png_bytep *row_pointers, int start_y, int end_y, int width) {
    int filter_size = 3, blur = 1;
    float filter[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    float sum = 0;
    for(int i = 0; i< filter_size; i++ ){
        for(int j = 0; j< filter_size; j++) {
            if(filter[i][j] < 0)
                blur = 0;
            sum += filter[i][j];
        }
    }
    if(blur){
    for (int i = 0; i < filter_size; i++)
        for (int j = 0; j < filter_size; j++) {
            filter[i][j] /= sum;
        }
    }
    for(int y = start_y; y < end_y; y++) {
        png_bytep row = row_pointers[y];
        #pragma omp parallel for default(none) shared(width, row, filter_size, filter, start_y, end_y, row_pointers, blur, y)
        for(int x = 0; x < width; x++) {
            int sum_R = 0, sum_G = 0, sum_B = 0, sum_A = 0;
            png_bytep px = &(row[x * 4]);
            int offset = filter_size / 2;
            for(int i = -offset; i <= offset; i++) {
                for (int j = -offset; j <= offset; j++) {
                    float curr_filter = filter[i + offset][j + offset];
                    // Mirror Edge Handling
                    int px_i = y + i >= start_y && y + i < end_y - 1 ? y + i : y - i;
                    int px_j = x + j >= 0 && x + j < width - 1 ? x + j : x - j;
                    // Wrap Edge Handling
                    // int px_i = y + i >= start_y ? (y + i < end_y - 1 ? y + i : 0 + i ) : end_y + i;
                    // int px_j = x + j >= 0 ? (x + j < width - 1 ? x + j : 0 + j) : width + j;
                    sum_R += (int) ((float) row_pointers[px_i][(px_j) * 4] * curr_filter);
                    sum_G += (int) ((float) row_pointers[px_i][(px_j) * 4 + 1] * curr_filter);
                    sum_B += (int) ((float) row_pointers[px_i][(px_j) * 4 + 2] * curr_filter);
                    sum_A += (int) ((float) row_pointers[px_i][(px_j) * 4 + 3] * curr_filter);
                }
            }
            if(!blur){
                sum_R = sum_R > 255 ? 255 : sum_R < 0 ? 0 : sum_R;
                sum_G = sum_G > 255 ? 255 : sum_G < 0 ? 0 : sum_G;
                sum_B = sum_B > 255 ? 255 : sum_B < 0 ? 0 : sum_B;
                sum_A = sum_A > 255 ? 255 : sum_A < 0 ? 0 : sum_A;
            }
            px[0] = sum_R, px[1] =sum_G, px[2] = sum_B, px[3] = sum_A;
        }
    }
}

unsigned char* png_matrix_flat(png_bytep *data, unsigned height, unsigned width){
    unsigned char* matrix;
    width *= 4;
    long matrix_size = height * width;
    if((matrix = (unsigned char*) malloc(matrix_size * sizeof(unsigned char))) == nullptr){
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
    for(long matrix_i = 0; matrix_i < matrix_size ; matrix_i++)
        matrix[matrix_i] = data[matrix_i / width][matrix_i % width];
    return matrix;
}

png_bytep* png_matrix_2d(unsigned char* matrix, unsigned height, unsigned width) {
    png_bytep* row_pointers;
    width *= 4;
    long matrix_size = height * width;
    if((row_pointers = (png_bytep*) malloc(height * sizeof(png_bytep))) == nullptr){
        std::cout<<"Malloc error"<<std::endl;
        exit(1);
    }
    for(int i = 0; i < height; i++)
        row_pointers[i] = (png_bytep) malloc(width * sizeof(unsigned char));

    for(int matrix_i = 0; matrix_i < matrix_size; matrix_i++){
        row_pointers[matrix_i / width][matrix_i % width] = matrix[matrix_i];
    }
    return row_pointers;
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int rank, size, root = 0;
    int *sendcounts, *displs;
    unsigned char* matrix;
    PNG_DATA data = {0, 0, '\0', '\0', nullptr};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == root){
        read_png_file(argv[1], &data);
        matrix = png_matrix_flat(data.row_pointers, data.height, data.width);
    }
    MPI_Bcast(&data.height, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    MPI_Bcast(&data.width, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);

    sendcounts =(int*) malloc(sizeof(int)*size);
    displs = (int*) malloc(sizeof(int)*size);
    int send_sum = 0, res = (int) data.height & size, size_per_process =  (int) data.height / size;
    for(int i = 0; i < size; i++){
        displs[i] = send_sum;
        sendcounts[i] = (i + 1 <= res) ? size_per_process + 1 : size_per_process;
        sendcounts[i] *= (int)data.width * 4;
        send_sum += sendcounts[i];
    }
    sendcounts[size - 1] -= 4 * (int)data.width;
    unsigned char *sub_matrix = (unsigned char*) malloc(sendcounts[rank] * sizeof(unsigned char*));
    MPI_Scatterv(matrix, sendcounts, displs, MPI_UNSIGNED_CHAR, sub_matrix, sendcounts[rank], MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

    int new_height = sendcounts[rank] / (int)( 4 * data.width);
    data.row_pointers = png_matrix_2d(sub_matrix, new_height, data.width);
    process_png_file(data.row_pointers, 0 , new_height , (int)data.width);
    sub_matrix = png_matrix_flat(data.row_pointers, new_height, data.width);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(sub_matrix, sendcounts[rank], MPI_UNSIGNED_CHAR, matrix, sendcounts, displs, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);
    MPI_Finalize();

    if(rank == root) {
        data.row_pointers = png_matrix_2d(matrix, data.height, data.width);
        write_png_file(argv[1] , &data);
    }
    return 0;
}
