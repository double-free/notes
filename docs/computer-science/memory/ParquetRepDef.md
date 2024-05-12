# Definition and Repetition Level in Parquet

在处理 parquet 格式的文件时，有两个抽象的定义：definition level 和 repetition level。对于刚接触这两个概念的人来讲，最需要理解的是，definition 和 repetition 到底是指谁的“定义”和“重复”？它对于 parquet 解析到底有什么作用？

本文通过查阅相关论文以及阅读 arrow-rs 的源码来解释这两个定义。

## Introduction

关于 definition level 和 repetition level 的作用，在文献 1 中有解释：

> A (repetition level, definition level) pair represents the **delta** between two **consecutive paths**.

即，如果我们把一个 field 用相对于 struct 的一个路径来描述，那这两个 levels 就是用于描述该 field 的路径。其中：

1. **repetition level 指定了连续两个 value 的 path 之间的共同前缀中 repeated field 的数量。注意，第一层级（即record 这个层级）也算是 repeated。**

2. **defition level 指定了路径中 optional 或 repeated 的 field 的数量。这其中不包括 record 层级。**

注意，以上两者都去掉了 required 的 field 的数量。因为它的路径是确定的。

在 parquet 里，field 有三类：

```rust
/// Representation of field types in schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum Repetition {
    /// Field is required (can not be null) and each record has exactly 1 value.
    REQUIRED,
    /// Field is optional (can be null) and each record has 0 or 1 values.
    OPTIONAL,
    /// Field is repeated and can contain 0 or more values.
    REPEATED,
}
```

显然，这其中 `REQUIRED` field 最简单，事实上，如果全是 required 的 field，实际上这两个 levels 并没有用武之地。


### Why do we need them?

为了让 parquet 支持嵌套的复杂数据结构， 即上面提到的 `OPTIONAL` 和 `REPEATED` 类型，我们引入了这两个 levels。

例如，我们需要存储一个 `Vec<Vec<i32>>`, 那我们实际用的是一个 `parquet::i32` 的 column，但其中存储了多个 `Vec<i32>`，这里我们称为一个 record。显然，一个 record 里包含了多个 `i32` value，我们需要一个编码方式将这些 value 组合起来。

### Example

举论文中的例子来讲，我们有如下 schema：

```text
message Document {
    required int64 DocId;
    optional group Links {
        repeated int64 Backward;
        repeated int64 Forward;
    }
    repeated group Name {
        repeated group Language {
            required string Code;
            optional string Country;
        }
        optional string Url;
    }
}
```

它有两条记录 `r1` 和 `r2`。

r1:

```text
DocId: 10
Links
    Forward: 20
    Forward: 40
    Forward: 60
Name
    Language
        Code: 'en-us'
        Country: 'us'
    Language
        Code: 'en'
    Url: 'http://A'
Name
    Url: 'http://B'
Name
    Language
        Code: 'en-gb'
        Country: 'gb'
```

r2:

```text
DocId: 20
Links
    Backward: 10
    Backward: 30
    Forward:  80
Name
    Url: 'http://C'
```

当我们存储这个数据的时候，由于是按列存储，我们一共有以下 6 个fields:

- `DocId`
- `Links.Forward`
- `Links.Backward`
- `Name.Url`
- `Name.Language.Code`
- `Name.Language.Country`

他们对应的 repetition level （简称 r）和 definition level （简称 d）分别为：

`DocId`: 均为 0，因为是 required field.

| value | r | d | path |
| :-: | :-: | :-: | :-: |
| 10 | 0 | 0 | `r1.DocId` |
| 20 | 0 | 0 | `r2.DocId` |

`Links.Forward`: 存在 `Links` 这个 optional field 和 `Forward` 这个 repeated field，因此 definition level 为 2。对于 repetition level，他们的路径分别为：

- `r1.Links.Forward1`
- `r1.Links.Forward2`
- `r1.Links.Forward3`
- `r2.Links.Forward1`

根据之前对 repetition level 的定义，这里的 `Links` 是 optional level，不能算。而 `Forward` 是最后一个 repetition field，已经被展开。因此只有 `r1` 是 common prefix，所以第二个第三个 repetition level 为 1.

| value | r | d | path |
| :-: | :-: | :-: | :-: |
| 20 | 0 | 2 | `r1.Links.Forward1` |
| 40 | 1 | 2 | `r1.Links.Forward2` |
| 60 | 1 | 2 | `r1.Links.Forward3` |
| 80 | 0 | 2 | `r2.Links.Forward1` |

`Links.Backward`: 关于 repetition 和 definition level 的计算和上面相似。唯一需要注意的是增加了一个 `NULL` entry。虽然 `r1` 中没有 `Links.Backward`，但是定义了 `Links`，为了保留这个信息，需要加入一个空值。

| value | r | d | path |
| :-: | :-: | :-: | :-: |
| NULL | 0 | 1 | `r1.Links` |
| 10 | 0 | 2 | `r2.Links.Backward1` |
| 30 | 1 | 2 | `r2.Links.Backward2` |

`Name.Url`: 第二个 path 为 `r1.Name2.Url`, 第三个空值 path 为 `r1.Name3`，因此它的 repetition level = 1，definition level 由于缺少了 Url 项，只有 1。

| value | r | d | path |
| :-: | :-: | :-: | :-: |
| "http://A" | 0 | 2 | `r1.Name1.Url` |
| "http://B" | 1 | 2 | `r1.Name2.Url` |
| NULL | 1 | 1 | `r1.Name3` |
| "http://C" | 0 | 2 | `r2.Name1.Url` |

`Name.Language.Code`: 第二个 path 为 `r1.Name1.Language2.Code`, 第三个为 `r1.Name2`, 注意 `Code` 是 required，因此不计入 definition level。

| value | r | d | path |
| :-: | :-: | :-: | :-: |
| "en-us" | 0 | 2 | `r1.Name1.Language1.Code` |
| "en" | 2 | 2 | `r1.Name1.Language2.Code` |
| NULL | 1 | 1 | `r1.Name2` |
| "en-gb" | 1 | 2 | `r1.Name3.Language1.Code` |
| NULL | 0 | 1 | `r2.Name1` |

`Name.Language.Country`: 推断方式同上。

| value | r | d | path | 
| :-: | :-: | :-: | :-: |
| "us" | 0 | 3 | `r1.Name1.Language1.Country` |
| NULL | 2 | 2 | `r1.Name1.Language2` |
| NULL | 1 | 1 | `r1.Name2` |
| "gb" | 1 | 3 | `r1.Name3.Language1.Country` |
| NULL | 0 | 1 | `r2.Name1` |

## Usage in `parquet` lib

Parquet 的 [read_records](https://docs.rs/parquet/latest/parquet/column/reader/struct.GenericColumnReader.html#method.read_records) 接口可以看出这两个 levels 的具体意义。

```rust
pub fn read_records(
    &mut self,
    max_records: usize,
    def_levels: Option<&mut D::Buffer>,
    rep_levels: Option<&mut R::Buffer>,
    values: &mut V::Buffer
) -> Result<(usize, usize, usize)>
```

这个函数会读取至多 `max_records` 条记录，每条记录都会 __完整__ 读取，这里的完整即下一个 repetition level 是 0。

这个函数的返回值是三个整数组成的 tuple，分别为：

1. 实际读取的总 records 数量 `n_records`
2. 非空的 value 数量 `n_valid_values`
3. 总共解析的 value 数量 `n_values`

注意，由于 record 可以存储数组等嵌套结构，record 的数量总是小于或等于 value 数量的。

接下来，根据注释：

> If the max definition level is 0, def_levels will be ignored and the number of records, non-null values and levels decoded will all be equal, otherwise def_levels will be populated with the number of levels read, with an error returned if it is None.

max definition level 如果是 0，则该 field 不允许有空值，必然有 `n_valid_values == n_values`。这里的参数 `def_levels` 是一个 `Option`，即如果已知这个列不允许为空，则可以不创建 `Vec<i16>` 来存储解析后的  definition level，否则需要提供一个 `Vec<i16>` 的引用，该函数会填充读到的 definition levels。取决于用户对需要读取的 parquet column 的先验知识：该列是否允许空值。


> If the max repetition level is 0, rep_levels will be ignored and the number of records and levels decoded will both be equal, otherwise rep_levels will be populated with the number of levels read, with an error returned if it is None.

max repetition level 如果是 0，则不存在嵌套类型的 field。必然有 `n_records == n_values`，即 records 数量等于 values 数量，但是并不保证这些 values 都非空值。跟 `def_levels` 类似， `rep_levels` 也是一个可选的参数，取决于用户对需要读取的 parquet column 的先验知识：该列是否是嵌套结构。

## Reference

1. [Dremel: Interactive Analysis of Web-Scale Datasets](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37217.pdf)
2. [Arrow and Parquet Part 2: Nested and Hierarchical Data using Structs and Lists](https://arrow.apache.org/blog/2022/10/08/arrow-parquet-encoding-part-2/)
3. [Wrapping one’s head around Repetition and Definition Levels in Dremel, powering BigQuery](https://akshays-blog.medium.com/wrapping-head-around-repetition-and-definition-levels-in-dremel-powering-bigquery-c1a33c9695da)
