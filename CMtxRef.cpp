#include "CMtxRef.h"

#define RAND_BORDER 5

using namespace MyRefAlgebra;

CMtx::CMtx(int columns, int rows, bool rand_init)
{
    if (columns <= 0 || rows <= 0) {
        m_columns = NULL;
        m_rows = NULL;
        m_matrix = nullptr;
        return;
    }

    m_matrix = new float* [columns];

    for (int i = 0; i < columns; i++) {
        m_matrix[i] = new float[rows];
        if (rand_init)
            for (int j = 0; j < rows; j++)
                m_matrix[i][j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * RAND_BORDER))) - RAND_BORDER;
        else
            for (int j = 0; j < rows; j++)
                m_matrix[i][j] = 0;
    }

    m_columns = columns;
    m_rows = rows;
}

CMtx::CMtx(const CMtx& rhs)
{
    copyFrom(rhs);
}

void CMtx::copyFrom(const CMtx& otherM)
{
    m_matrix = new float* [otherM.m_columns];

    for (int i = 0; i < otherM.m_columns; i++)
        m_matrix[i] = new float[otherM.m_rows];

    m_columns = otherM.m_columns;
    m_rows = otherM.m_rows;

    for (int x = 0; x < m_columns; x++)
        for (int y = 0; y < m_rows; y++)
            m_matrix[x][y] = otherM.m_matrix[x][y];

    if (DEBUG) std::cout << "copy" << std::endl;
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
    if (DEBUG) std::cout << "move" << std::endl;
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

CMtx CMtx::operator+(const CMtx& otherM) const
{
    if (!equalSize(otherM)) throw std::invalid_argument("Sizes aren't equal");

    CMtx result(m_columns, m_rows);
    for (int x = 0; x < m_columns; x++)
        for (int y = 0; y < m_rows; y++)
            result.m_matrix[x][y] = this->m_matrix[x][y] + otherM.m_matrix[x][y];

    return std::move(result); //??
}

CMtx CMtx::operator-(const CMtx& otherM) const
{
    if (!equalSize(otherM)) throw std::invalid_argument("Sizes aren't equal");

    CMtx result(m_columns, m_rows);
    for (int x = 0; x < m_columns; x++)
        for (int y = 0; y < m_rows; y++)
            result.m_matrix[x][y] = this->m_matrix[x][y] - otherM.m_matrix[x][y];

    return std::move(result);
}

CMtx CMtx::operator*(float number) const
{
    CMtx result(m_columns, m_rows);
    for (int x = 0; x < m_columns; x++)
        for (int y = 0; y < m_rows; y++)
            result.m_matrix[x][y] = this->m_matrix[x][y] * number;

    return std::move(result);
}

CMtx CMtx::operator*(const CMtx& otherM) const
{
    if (this->m_columns != otherM.m_rows) { throw std::invalid_argument("You can't multiply those matrixes!"); }
    CMtx result(otherM.m_columns, m_rows);
    float res = 0;

    int col = result.m_columns;
    int rows = result.m_rows;

    for (int x = 0; x < col; x++)
        for (int y = 0; y < rows; y++) {
            res = 0;
            for (int i = 0; i < this->m_columns; i++)
                res += this->m_matrix[i][y] * otherM.m_matrix[x][i];

            result.m_matrix[x][y] = res;
        }

    return std::move(result);
}

void CMtx::operator=(const CMtx& otherM)
{
    if (this == &otherM) return;
    deleteMatrix();
    copyFrom(otherM);
}

void CMtx::operator=(CMtx&& otherM)
{
    if (this == &otherM) return;
    deleteMatrix();
    moveFrom(std::move(otherM));
}

CMtx CMtx::operator-() const
{
    CMtx result(m_columns, m_rows);
    for (int x = 0; x < m_columns; x++)
        for (int y = 0; y < m_rows; y++)
            result.m_matrix[x][y] = this->m_matrix[x][y] * -1;

    return std::move(result);
}

CMtx CMtx::operator~() const
{
    CMtx result(m_rows, m_columns);

    for (int x = 0; x < m_columns; x++)
        for (int y = 0; y < m_rows; y++)
            result.m_matrix[y][x] = this->m_matrix[x][y];

    return std::move(result);
}
