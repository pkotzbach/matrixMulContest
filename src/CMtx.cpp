#include "CMtx.h"

using namespace MyAlgebra;

const float CMtx::ALG_PRECISION = 0.001;
#define RAND_BORDER 5

void CMtx::createPrivates(int columns, int rows)
{
    m_matrix = new float* [columns];

    for (int i = 0; i < columns; i++) {
        m_matrix[i] = new float[rows];
    }

    m_columns = columns;
    m_rows = rows;
}

CMtx::CMtx(int columns, int rows, bool rand_init)
{
    if (columns <= 0 || rows <= 0) {
        m_columns = NULL;
        m_rows = NULL;
        m_matrix = nullptr;
        return;
    }

    createPrivates(columns, rows);

    if (rand_init) fillRandom();
    else fillWith(0);
}

CMtx::CMtx(const CMtx& rhs)
{
    copyFrom(rhs);
}

void CMtx::copyFrom(const CMtx& otherM)
{
    createPrivates(otherM.m_columns, otherM.m_rows);

    for (int x = 0; x < m_columns; x++)
        for (int y = 0; y < m_rows; y++)
            m_matrix[x][y] = otherM.m_matrix[x][y];
}

CMtx::CMtx(CMtx&& otherM)
{
    moveFrom(std::move(otherM));
}

void CMtx::moveFrom(CMtx&& otherM)
{
    m_matrix = otherM.m_matrix;
    m_columns = otherM.m_columns;
    m_rows = otherM.m_rows;

    otherM.m_matrix = nullptr;
}

CMtx::CMtx(int rows, float diagonal)
{
    if (rows <= 0) {
        m_columns = NULL;
        m_rows = NULL;
        m_matrix = nullptr;
        return;
    }
    createPrivates(rows, rows);

    for (int x = 0; x < m_columns; x++) {
        for (int y = 0; y < m_rows; y++) {
            m_matrix[x][y] = diagonal * (x == y);
        }
    }
}

CMtx::~CMtx()
{
    deleteMatrix();
}

void CMtx::deleteMatrix()
{
    if (m_matrix == nullptr)
        return;

    for (int i = 0; i < m_columns; i++)
        delete[] m_matrix[i];

    delete[] m_matrix;
}

void CMtx::display() const
{
    for (int y = 0; y < m_rows; y++) {
        for (int x = 0; x < m_columns; x++)
            std::cout << m_matrix[x][y] << " ";
        std::cout << "\n";
    }
}

float* CMtx::operator[](int row_ind)
{
    return m_matrix[row_ind];
}

void CMtx::addSIMD(const CMtx& otherM, CMtx& result, int x1, int y1, int x2, int y2) const
{
    assert(y1 % IN_SIMD == 0);
    assert(y2 % IN_SIMD == 0);

    float* r_it, * a_it, * b_it;

    for (int x = x1; x < x2; x++) {
        r_it = result.m_matrix[x];
        a_it = m_matrix[x];
        b_it = otherM.m_matrix[x];
            for (int y = y1; y < y2; y += IN_SIMD) {
                _mm256_store_ps(&r_it[y], _mm256_add_ps(_mm256_load_ps(&a_it[y]), _mm256_load_ps(&b_it[y])));
            }
        }
}

CMtx CMtx::operator+(const CMtx& otherM) const
{
    if (!equalSize(otherM)) throw std::invalid_argument("Sizes aren't equal");

    CMtx result(m_columns, m_rows, false);

    int N_y = m_rows / IN_SIMD * IN_SIMD;

    addSIMD(otherM, result, 0, 0, m_columns, N_y);

    if (N_y < m_rows) {
        for (int x = 0; x < m_columns; x++)
            for (int y = N_y; y < m_rows; y++)
                result.m_matrix[x][y] = this->m_matrix[x][y] + otherM.m_matrix[x][y];
    }

    return std::move(result);
}

void CMtx::subSIMD(const CMtx& otherM, CMtx& result, int x1, int y1, int x2, int y2) const
{
    assert(y1 % IN_SIMD == 0);
    assert(y2 % IN_SIMD == 0);

    float* r_it, * a_it, * b_it;

    for (int x = x1; x < x2; x++) {
        r_it = result.m_matrix[x];
        a_it = m_matrix[x];
        b_it = otherM.m_matrix[x];
        for (int y = y1; y < y2; y += IN_SIMD) {
            _mm256_store_ps(&r_it[y], _mm256_sub_ps(_mm256_load_ps(&a_it[y]), _mm256_load_ps(&b_it[y])));
        }
    }
}

CMtx CMtx::operator-(const CMtx& otherM) const
{
    if (!equalSize(otherM)) throw std::invalid_argument("Sizes aren't equal");

    CMtx result(m_columns, m_rows, false);

    int N_y = m_rows / IN_SIMD * IN_SIMD;

    subSIMD(otherM, result, 0, 0, m_columns, N_y);

    if (N_y < m_rows) {
        for (int x = 0; x < m_columns; x++)
            for (int y = N_y; y < m_rows; y++)
                result.m_matrix[x][y] = this->m_matrix[x][y] - otherM.m_matrix[x][y];
    }

    return std::move(result);
}

void CMtx::mulScalarSIMD(float number, CMtx& result, int x1, int y1, int x2, int y2) const
{
    assert(y1 % IN_SIMD == 0);
    assert(y2 % IN_SIMD == 0);

    const int exp_val = 4;

    const int N_x = result.m_columns / exp_val * exp_val;
    __m256 num = _mm256_set1_ps(number);;
    float* r_it[exp_val], * a_it[exp_val];

    for (int x = x1; x < N_x; x += exp_val) {
        r_it[0] = result.m_matrix[x];
        r_it[1] = result.m_matrix[x + 1];
        r_it[2] = result.m_matrix[x + 2];
        r_it[3] = result.m_matrix[x + 3];
        a_it[0] = m_matrix[x];
        a_it[1] = m_matrix[x + 1];
        a_it[2] = m_matrix[x + 2];
        a_it[3] = m_matrix[x + 3];
        for (int y = y1; y < y2; y += IN_SIMD) {
            _mm256_store_ps(&r_it[0][y], _mm256_mul_ps(_mm256_load_ps(&a_it[0][y]), num));
            _mm256_store_ps(&r_it[1][y], _mm256_mul_ps(_mm256_load_ps(&a_it[1][y]), num));
            _mm256_store_ps(&r_it[2][y], _mm256_mul_ps(_mm256_load_ps(&a_it[2][y]), num));
            _mm256_store_ps(&r_it[3][y], _mm256_mul_ps(_mm256_load_ps(&a_it[3][y]), num));
        }
    }

    for (int x = N_x; x < x2; x++) {
        r_it[0] = result.m_matrix[x];
        a_it[0] = m_matrix[x];
        for (int y = y1; y < y2; y += IN_SIMD) {
            _mm256_store_ps(&r_it[0][y], _mm256_mul_ps(_mm256_load_ps(&a_it[0][y]), num));
        }
    }
}

CMtx CMtx::operator*(float number) const
{
    CMtx result(m_columns, m_rows, false);

    int N_y = m_rows / IN_SIMD * IN_SIMD;

    mulScalarSIMD(number, result, 0, 0, m_columns, N_y);

    if (N_y < m_rows) {
        for (int x = 0; x < m_columns; x++)
            for (int y = N_y; y < m_rows; y++)
                result.m_matrix[x][y] = this->m_matrix[x][y] * number;
    }

    return std::move(result);
}

void CMtx::mulTilesSIMD(const CMtx& otherM, CMtx& result, int x1, int y1, int x2, int y2, const int N_mul) const
{
    assert(x1 % T_x == 0);
    assert(x2 % T_x == 0);
    assert(y1 % T_y == 0);
    assert(y2 % T_y == 0);

    int bound_x = 0;
    int bound_y = 0;

    const int blocking_val = 4;
    float* a_it;
    float* r_it[blocking_val];

    __m256 a, c, b[blocking_val];

    //A * B = C
    //this * otherM = result

    for (int C_y = y1; C_y < y2; C_y += T_y)
        for (int C_x = x1; C_x < x2; C_x += T_x) {
            bound_x = C_x + T_x;
            bound_y = C_y + T_y;
            for (int mul_it = 0; mul_it < N_mul; mul_it += T_i)
                for (int x = C_x; x < bound_x; x += blocking_val) {
                    r_it[0] = result.m_matrix[x];
                    r_it[1] = result.m_matrix[x + 1];
                    r_it[2] = result.m_matrix[x + 2];
                    r_it[3] = result.m_matrix[x + 3];
                    for (int i = 0; i < T_i; i++) {
                        a_it = m_matrix[mul_it + i];
                        b[0] = _mm256_set1_ps(otherM.m_matrix[x][mul_it + i]);
                        b[1] = _mm256_set1_ps(otherM.m_matrix[x + 1][mul_it + i]);
                        b[2] = _mm256_set1_ps(otherM.m_matrix[x + 2][mul_it + i]);
                        b[3] = _mm256_set1_ps(otherM.m_matrix[x + 3][mul_it + i]);
                        for (int y = C_y; y < bound_y; y += IN_SIMD) {
                            a = _mm256_load_ps(&a_it[y]);

                            c = _mm256_load_ps(&r_it[0][y]);
                            c = _mm256_add_ps(_mm256_mul_ps(a, b[0]), c);
                            _mm256_store_ps(&r_it[0][y], c);

                            c = _mm256_load_ps(&r_it[1][y]);
                            c = _mm256_add_ps(_mm256_mul_ps(a, b[1]), c);
                            _mm256_store_ps(&r_it[1][y], c);

                            c = _mm256_load_ps(&r_it[2][y]);
                            c = _mm256_add_ps(_mm256_mul_ps(a, b[2]), c);
                            _mm256_store_ps(&r_it[2][y], c);

                            c = _mm256_load_ps(&r_it[3][y]);
                            c = _mm256_add_ps(_mm256_mul_ps(a, b[3]), c);
                            _mm256_store_ps(&r_it[3][y], c);
                        }
                    }
                }
        }
}

void CMtx::mulBigMatrix(const CMtx& otherM, CMtx& result) const
{
#ifdef DEBUG
    std::cout << "mulBigMatrix" << std::endl;
#endif // DEBUG

    //sizes for tiling
    const int N_y = result.m_rows / T_y * T_y;
    const int N_x = result.m_columns / T_x * T_x;
    const int N_y_by_2 = (result.m_rows / 2) / T_y * T_y;
    const int N_x_by_2 = (result.m_columns / 2) / T_x * T_x;
    const int N_mul = m_columns / T_i * T_i;

    //A * B = C
    //this * otherM = result

    //std::thread thr1(&MyAlgebra::CMtx::mulTilesSIMD, this, std::ref(otherM), std::ref(result), N_x_by_2, 0, N_x, N_y); //1 cwiartka
    //std::thread thr2(&MyAlgebra::CMtx::mulTilesSIMD, this, std::ref(otherM), std::ref(result), 0, 0, N_x_by_2, N_y); //2 cwiartka

    std::thread thr1(&MyAlgebra::CMtx::mulTilesSIMD, this, std::ref(otherM), std::ref(result), N_x_by_2, 0, N_x, N_y_by_2, N_mul); //1 cwiartka
    std::thread thr2(&MyAlgebra::CMtx::mulTilesSIMD, this, std::ref(otherM), std::ref(result), 0, 0, N_x_by_2, N_y_by_2, N_mul); //2 cwiartka
    std::thread thr3(&MyAlgebra::CMtx::mulTilesSIMD, this, std::ref(otherM), std::ref(result), 0, N_y_by_2, N_x_by_2, N_y, N_mul); //3 cwiartka
    std::thread thr4(&MyAlgebra::CMtx::mulTilesSIMD, this, std::ref(otherM), std::ref(result), N_x_by_2, N_y_by_2, N_x, N_y, N_mul); //4 cwiartka

    thr1.join();
    thr2.join();
    thr3.join();
    thr4.join();

    __m256 a, b, c;
    float* r_it, * a_it;

    //N_x
    if (N_x < result.m_columns) {
        for (int x = N_x; x < result.m_columns; x++) {
            r_it = result.m_matrix[x];
            for (int i = 0; i < this->m_columns; i++) {
                a_it = m_matrix[i];
                b = _mm256_set1_ps(otherM.m_matrix[x][i]);
                for (int y = 0; y < N_y; y += IN_SIMD) {
                    //result.m_matrix[x][y] += this->m_matrix[i][y] * otherM.m_matrix[x][i];
                    _mm256_store_ps(&r_it[y], _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(&a_it[y]), b), _mm256_load_ps(&r_it[y])));
                }
            }
        }
    }

    float b_comp;

    //N_y and duplication
    if (N_y < result.m_rows) {
        for (int x = 0; x < result.m_columns; x++) {
            r_it = result.m_matrix[x];
            for (int i = 0; i < this->m_columns; i++) {
                a_it = this->m_matrix[i];
                b_comp = otherM.m_matrix[x][i];
                for (int y = N_y; y < result.m_rows; y++)
                    r_it[y] += a_it[y] * b_comp;
            }
        }
    }

    //N_mul
    if (N_mul < m_columns) {
        for (int x = 0; x < N_x; x++) {
            r_it = result.m_matrix[x];
            for (int i = N_mul; i < this->m_columns; i++) {
                a_it = m_matrix[i];
                b = _mm256_set1_ps(otherM.m_matrix[x][i]);
                for (int y = 0; y < N_y; y += IN_SIMD) { //N_y jest podzielne przez 8
                    //result.m_matrix[x][y] += this->m_matrix[i][y] * otherM.m_matrix[x][i];
                    _mm256_store_ps(&r_it[y], _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(&a_it[y]), b), _mm256_load_ps(&r_it[y])));
                }
            }
        }
    }
}

void CMtx::mulMediumMatrix(const CMtx& otherM, CMtx& result) const
{
#ifdef DEBUG
    std::cout << "mulMediumMatrix" << std::endl;
#endif // DEBUG

    //sizes for tiling
    const int N_y = result.m_rows / T_y * T_y;
    const int N_x = result.m_columns / T_x * T_x;
    const int N_mul = m_columns / T_i * T_i;

    mulTilesSIMD(otherM, result, 0, 0, N_x, N_y, N_mul);

    __m256 a, b, c;
    float* r_it, * a_it;

    //N_x
    if (N_x < result.m_columns) {
        for (int x = N_x; x < result.m_columns; x++) {
            r_it = result.m_matrix[x];
            for (int i = 0; i < this->m_columns; i++) {
                a_it = m_matrix[i];
                b = _mm256_set1_ps(otherM.m_matrix[x][i]);
                for (int y = 0; y < N_y; y += IN_SIMD) {
                    //result.m_matrix[x][y] += this->m_matrix[i][y] * otherM.m_matrix[x][i];
                    _mm256_store_ps(&r_it[y], _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(&a_it[y]), b), _mm256_load_ps(&r_it[y])));
                }
            }
        }
    }

    float b_comp;

    //N_y and duplication
    if (N_y < result.m_rows) {
        for (int x = 0; x < result.m_columns; x++) {
            r_it = result.m_matrix[x];
            for (int i = 0; i < this->m_columns; i++) {
                a_it = this->m_matrix[i];
                b_comp = otherM.m_matrix[x][i];
                for (int y = N_y; y < result.m_rows; y++)
                    r_it[y] += a_it[y] * b_comp;
            }
        }
    }

    //N_mul
    if (N_mul < m_columns) {
        for (int x = 0; x < N_x; x++) {
            r_it = result.m_matrix[x];
            for (int i = N_mul; i < this->m_columns; i++) {
                a_it = m_matrix[i];
                b = _mm256_set1_ps(otherM.m_matrix[x][i]);
                for (int y = 0; y < N_y; y += IN_SIMD) {
                    //result.m_matrix[x][y] += this->m_matrix[i][y] * otherM.m_matrix[x][i];
                    _mm256_store_ps(&r_it[y], _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(&a_it[y]), b), _mm256_load_ps(&r_it[y])));
                }
            }
        }
    }
}

void CMtx::mulSmallMatrix(const CMtx& otherM, CMtx& result) const
{
#ifdef DEBUG
    std::cout << "mulSmallMatrix" << std::endl;
#endif // DEBUG
    assert(result.m_rows >= IN_SIMD);

    const int blocking_val = 4;
    float* a_it;
    float* r_it[blocking_val];
    const int N_y = result.m_rows / IN_SIMD * IN_SIMD;
    const int N_x = result.m_columns / blocking_val * blocking_val;
    __m256 a, c, b[blocking_val];

    for (int x = 0; x < N_x; x += blocking_val) {
        r_it[0] = result.m_matrix[x];
        r_it[1] = result.m_matrix[x + 1];
        r_it[2] = result.m_matrix[x + 2];
        r_it[3] = result.m_matrix[x + 3];
        for (int i = 0; i < m_columns; i++) {
            a_it = m_matrix[i];
            b[0] = _mm256_set1_ps(otherM.m_matrix[x][i]);
            b[1] = _mm256_set1_ps(otherM.m_matrix[x + 1][i]);
            b[2] = _mm256_set1_ps(otherM.m_matrix[x + 2][i]);
            b[3] = _mm256_set1_ps(otherM.m_matrix[x + 3][i]);
            for (int y = 0; y < N_y; y += IN_SIMD) {
                a = _mm256_load_ps(&a_it[y]);

                c = _mm256_load_ps(&r_it[0][y]);
                c = _mm256_add_ps(_mm256_mul_ps(a, b[0]), c);
                _mm256_store_ps(&r_it[0][y], c);

                c = _mm256_load_ps(&r_it[1][y]);
                c = _mm256_add_ps(_mm256_mul_ps(a, b[1]), c);
                _mm256_store_ps(&r_it[1][y], c);

                c = _mm256_load_ps(&r_it[2][y]);
                c = _mm256_add_ps(_mm256_mul_ps(a, b[2]), c);
                _mm256_store_ps(&r_it[2][y], c);

                c = _mm256_load_ps(&r_it[3][y]);
                c = _mm256_add_ps(_mm256_mul_ps(a, b[3]), c);
                _mm256_store_ps(&r_it[3][y], c);
            }
        }

    }

    if (N_x < result.m_columns) {
        for (int x = N_x; x < result.m_columns; x++) {
            r_it[0] = result.m_matrix[x];
            for (int i = 0; i < m_columns; i++) {
                a_it = m_matrix[i];
                b[0] = _mm256_set1_ps(otherM.m_matrix[x][i]);
                for (int y = 0; y < N_y; y += IN_SIMD) {
                    //result.m_matrix[x][y] += this->m_matrix[i][y] * otherM.m_matrix[x][i];
                    _mm256_store_ps(&r_it[0][y], _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(&a_it[y]), b[0]), _mm256_load_ps(&r_it[0][y])));
                }
            }
        }
    }

    float b_comp;
    if (N_y < result.m_rows) {
        for (int x = 0; x < result.m_columns; x++) {
            r_it[0] = result.m_matrix[x];
            for (int i = 0; i < m_columns; i++) {
                a_it = this->m_matrix[i];
                b_comp = otherM.m_matrix[x][i];
                for (int y = N_y; y < result.m_rows; y++)
                    r_it[0][y] += a_it[y] * b_comp;
            }
        }
    }

}

void CMtx::mulMatrixIKJ(const CMtx& otherM, CMtx& result) const
{
#ifdef DEBUG
    std::cout << "mulMatrixIKJ" << std::endl;
#endif // DEBUG
    float b_comp, * r_it, * a_it;

    for (int x = 0; x < result.m_columns; x++) {
        r_it = result.m_matrix[x];
        for (int i = 0; i < this->m_columns; i++) {
            a_it = this->m_matrix[i];
            b_comp = otherM.m_matrix[x][i];
            for (int y = 0; y < result.m_rows; y++)
                r_it[y] += a_it[y] * b_comp;
        }
    }
}

CMtx CMtx::operator*(const CMtx& otherM) const
{
    if (this->m_columns != otherM.m_rows) throw std::invalid_argument("You can't multiply those matrixes!");
    CMtx result(otherM.m_columns, m_rows, false);
    const int matrix_size = result.m_columns * result.m_rows;

    if (result.m_rows < IN_SIMD)                                                mulMatrixIKJ(otherM, result);
    else if (matrix_size < MEDIUM_MATRIX_BOUND ||
             result.m_columns < T_x || result.m_rows < T_y || m_columns < T_i)  mulSmallMatrix(otherM, result);
    else if (matrix_size < BIG_MATRIX_BOUND)                                    mulMediumMatrix(otherM, result);
    else                                                                        mulBigMatrix(otherM, result);

    return std::move(result);
}

void CMtx::operator=(const CMtx& otherM)
{
    if (this == &otherM) return;
    deleteMatrix();
    copyFrom(otherM);
}

void CMtx::operator=(const float diagonal)
{
    if (m_columns != m_rows) throw std::exception("You can't make this matrix diagonal!");
    
    for (int x = 0; x < m_columns; x++) {
        for (int y = 0; y < m_rows; y++) {
            m_matrix[x][y] = diagonal * (x == y);
        }
    }
}

void CMtx::operator=(CMtx&& otherM)
{
    if (this == &otherM) return;
    deleteMatrix();
    moveFrom(std::move(otherM));
}

CMtx CMtx::operator-() const
{
    CMtx result(m_columns, m_rows, false);

    int N_y = m_rows / IN_SIMD * IN_SIMD;

    mulScalarSIMD(-1, result, 0, 0, m_columns, N_y);

    if (N_y < m_rows) {
        for (int x = 0; x < m_columns; x++)
            for (int y = N_y; y < m_rows; y++)
                result.m_matrix[x][y] = this->m_matrix[x][y] * -1;
    }

    return std::move(result);
}

CMtx CMtx::operator~() const
{
    CMtx result(m_rows, m_columns, false);

    for (int x = 0; x < m_columns; x++)
        for (int y = 0; y < m_rows; y++)
            result.m_matrix[y][x] = this->m_matrix[x][y];

    return std::move(result);
}

CMtx CMtx::operator^(int power) const
{
    if (power < 0) throw std::exception("Power can't be negative number!");

    if (power == 0) {
        if (m_columns != m_rows) throw std::exception("You can't make this matrix diagonal!");
        CMtx result(m_columns, 1.0f);
        return std::move(result);
    }

    if (power > 0) {
        CMtx result(*this);
        if (power == 1) return std::move(result);

        for (int i = 1; i < power; i++) {
            result = result * *this;
        }
        return std::move(result);
    }
}

bool CMtx::operator==(const CMtx&& otherM) const
{
    if (m_columns != otherM.m_columns || m_rows != otherM.m_rows) return false;

    for (int x = 0; x < m_columns; x++)
        for (int y = 0; y < m_rows; y++)
            if (abs(m_matrix[x][y] - otherM.m_matrix[x][y]) > ALG_PRECISION) return false;

    return true;
}


void CMtx::fillWith(int number)
{
    int N_y = m_rows / IN_SIMD * IN_SIMD;

    __m256 num = _mm256_set1_ps(number);;
    float* r_it;

    for (int x = 0; x < m_columns; x++) {
        r_it = m_matrix[x];
        for (int y = 0; y < N_y; y += IN_SIMD) {
            _mm256_store_ps(&r_it[y], num);
        }
    }

    if (N_y < m_rows) {
        for (int x = 0; x < m_columns; x++)
            for (int y = N_y; y < m_rows; y++)
                m_matrix[x][y] = number;
    }
}

void CMtx::fillRandom()
{
    for (int x = 0; x < m_columns; x++) {
        for (int y = 0; y < m_rows; y++) {
            m_matrix[x][y] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * RAND_BORDER))) - RAND_BORDER;
        }
    }
}

CMtx MyAlgebra::operator*(float multiplier, const CMtx& rhs)
{
    return std::move(rhs * multiplier);
}