//! # Datalog IR
//!
//! Intermediate Representation types for Datalog query plans.
//!
//! This crate provides a canonical IR definition for representing Datalog query
//! execution plans, suitable for use in query optimizers and execution engines.
//!
//! ## Overview
//!
//! The main types are:
//! - [`IRNode`] - Query plan operators (Scan, Map, Filter, Join, Distinct, Union)
//! - [`Predicate`] - Filter conditions with support for comparisons and logical operators
//!
//! ## Example
//!
//! ```rust
//! use datalog_ir::{IRNode, Predicate};
//!
//! // Create a scan of the "edge" relation
//! let scan = IRNode::Scan {
//!     relation: "edge".to_string(),
//!     schema: vec!["x".to_string(), "y".to_string()],
//! };
//!
//! // Add a filter: x > 5
//! let filtered = IRNode::Filter {
//!     input: Box::new(scan),
//!     predicate: Predicate::ColumnGtConst(0, 5),
//! };
//!
//! // Get the output schema
//! assert_eq!(filtered.output_schema(), vec!["x", "y"]);
//! ```

use std::collections::HashSet;

// ============================================================================
// IR Node Types
// ============================================================================

/// Represents an operator in the query plan.
///
/// `IRNode` is the core building block for constructing Datalog query plans.
/// Each variant represents a different relational algebra operation.
///
/// # Example
///
/// ```rust
/// use datalog_ir::{IRNode, Predicate};
///
/// // Build a simple query plan: scan -> filter -> project
/// let plan = IRNode::Map {
///     input: Box::new(IRNode::Filter {
///         input: Box::new(IRNode::Scan {
///             relation: "users".to_string(),
///             schema: vec!["id".to_string(), "name".to_string(), "age".to_string()],
///         }),
///         predicate: Predicate::ColumnGtConst(2, 18), // age > 18
///     }),
///     projection: vec![1], // Keep only "name"
///     output_schema: vec!["name".to_string()],
/// };
///
/// assert_eq!(plan.output_schema(), vec!["name"]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IRNode {
    /// Scan a relation (read from EDB or IDB).
    ///
    /// This is typically the leaf node in a query plan, representing
    /// a base table or derived relation.
    Scan {
        /// Name of the relation to scan.
        relation: String,
        /// Column names in the relation's schema.
        schema: Vec<String>,
    },

    /// Map (project/transform columns).
    ///
    /// Selects a subset of columns from the input and optionally reorders them.
    Map {
        /// The input node to project from.
        input: Box<IRNode>,
        /// Indices of input columns to keep (in output order).
        projection: Vec<usize>,
        /// Names of columns in the output schema.
        output_schema: Vec<String>,
    },

    /// Filter (select rows matching a predicate).
    ///
    /// Passes through only rows where the predicate evaluates to true.
    /// The schema is unchanged from the input.
    Filter {
        /// The input node to filter.
        input: Box<IRNode>,
        /// The condition rows must satisfy.
        predicate: Predicate,
    },

    /// Join two inputs on shared keys.
    ///
    /// Performs an equi-join on one or more key columns.
    /// The output schema is the concatenation of left and right schemas.
    Join {
        /// Left input to the join.
        left: Box<IRNode>,
        /// Right input to the join.
        right: Box<IRNode>,
        /// Column indices from left input to join on.
        left_keys: Vec<usize>,
        /// Column indices from right input to join on.
        right_keys: Vec<usize>,
        /// Names of columns in the joined output.
        output_schema: Vec<String>,
    },

    /// Distinct (remove duplicate rows).
    ///
    /// Returns only unique rows from the input.
    Distinct {
        /// The input node to deduplicate.
        input: Box<IRNode>,
    },

    /// Union (combine multiple inputs).
    ///
    /// Concatenates rows from all inputs. All inputs must have the same schema.
    Union {
        /// The input nodes to combine.
        inputs: Vec<IRNode>,
    },
}

impl IRNode {
    /// Returns the output schema of this node.
    ///
    /// The schema represents the column names that will be present in the
    /// output of this operator. For `Filter` and `Distinct`, the schema
    /// passes through unchanged from the input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use datalog_ir::IRNode;
    ///
    /// let scan = IRNode::Scan {
    ///     relation: "edge".to_string(),
    ///     schema: vec!["x".to_string(), "y".to_string()],
    /// };
    /// assert_eq!(scan.output_schema(), vec!["x", "y"]);
    /// ```
    pub fn output_schema(&self) -> Vec<String> {
        match self {
            IRNode::Scan { schema, .. } => schema.clone(),
            IRNode::Map { output_schema, .. } => output_schema.clone(),
            IRNode::Filter { input, .. } => input.output_schema(),
            IRNode::Join { output_schema, .. } => output_schema.clone(),
            IRNode::Distinct { input } => input.output_schema(),
            IRNode::Union { inputs } => {
                if inputs.is_empty() {
                    vec![]
                } else {
                    inputs[0].output_schema()
                }
            }
        }
    }

    /// Pretty prints the IR tree for debugging.
    ///
    /// Returns a formatted string representation of the query plan tree,
    /// with each level indented by the specified amount.
    ///
    /// # Arguments
    ///
    /// * `indent` - The base indentation level (number of 2-space units).
    ///
    /// # Example
    ///
    /// ```rust
    /// use datalog_ir::{IRNode, Predicate};
    ///
    /// let plan = IRNode::Filter {
    ///     input: Box::new(IRNode::Scan {
    ///         relation: "edge".to_string(),
    ///         schema: vec!["x".to_string(), "y".to_string()],
    ///     }),
    ///     predicate: Predicate::ColumnGtConst(0, 5),
    /// };
    ///
    /// let output = plan.pretty_print(0);
    /// assert!(output.contains("Filter"));
    /// assert!(output.contains("Scan"));
    /// ```
    pub fn pretty_print(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);

        match self {
            IRNode::Scan { relation, schema } => {
                format!("{}Scan({}) schema={:?}", prefix, relation, schema)
            }
            IRNode::Map {
                input,
                projection,
                output_schema,
            } => {
                format!(
                    "{}Map(projection={:?}, output={:?})\n{}",
                    prefix,
                    projection,
                    output_schema,
                    input.pretty_print(indent + 1)
                )
            }
            IRNode::Filter { input, predicate } => {
                format!(
                    "{}Filter({:?})\n{}",
                    prefix,
                    predicate,
                    input.pretty_print(indent + 1)
                )
            }
            IRNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                output_schema,
            } => {
                format!(
                    "{}Join(left_keys={:?}, right_keys={:?}, output={:?})\n{}\n{}",
                    prefix,
                    left_keys,
                    right_keys,
                    output_schema,
                    left.pretty_print(indent + 1),
                    right.pretty_print(indent + 1)
                )
            }
            IRNode::Distinct { input } => {
                format!("{}Distinct\n{}", prefix, input.pretty_print(indent + 1))
            }
            IRNode::Union { inputs } => {
                let mut result = format!("{}Union\n", prefix);
                for input in inputs {
                    result.push_str(&input.pretty_print(indent + 1));
                    result.push('\n');
                }
                result
            }
        }
    }

    /// Returns `true` if this node is a [`Scan`](IRNode::Scan).
    #[inline]
    pub fn is_scan(&self) -> bool {
        matches!(self, IRNode::Scan { .. })
    }

    /// Returns `true` if this node is a [`Join`](IRNode::Join).
    #[inline]
    pub fn is_join(&self) -> bool {
        matches!(self, IRNode::Join { .. })
    }
}

// ============================================================================
// Predicate Types
// ============================================================================

/// Predicate for filtering rows in [`IRNode::Filter`].
///
/// Predicates can express column comparisons with constants, column-to-column
/// comparisons, and logical combinations using `And` and `Or`.
///
/// # Example
///
/// ```rust
/// use datalog_ir::Predicate;
///
/// // Simple comparison: column 0 > 10
/// let pred = Predicate::ColumnGtConst(0, 10);
///
/// // Compound predicate: (col0 > 10) AND (col1 = col2)
/// let compound = Predicate::And(
///     Box::new(Predicate::ColumnGtConst(0, 10)),
///     Box::new(Predicate::ColumnsEq(1, 2)),
/// );
///
/// // Find which columns are referenced
/// let cols = compound.referenced_columns();
/// assert!(cols.contains(&0));
/// assert!(cols.contains(&1));
/// assert!(cols.contains(&2));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Predicate {
    /// Column equals constant: `column[idx] == value`.
    ColumnEqConst(usize, i64),
    /// Column not equals constant: `column[idx] != value`.
    ColumnNeConst(usize, i64),
    /// Column greater than constant: `column[idx] > value`.
    ColumnGtConst(usize, i64),
    /// Column less than constant: `column[idx] < value`.
    ColumnLtConst(usize, i64),
    /// Column greater or equal to constant: `column[idx] >= value`.
    ColumnGeConst(usize, i64),
    /// Column less or equal to constant: `column[idx] <= value`.
    ColumnLeConst(usize, i64),
    /// Two columns are equal: `column[left] == column[right]`.
    ColumnsEq(usize, usize),
    /// Two columns are not equal: `column[left] != column[right]`.
    ColumnsNe(usize, usize),
    /// Logical AND of two predicates.
    And(Box<Predicate>, Box<Predicate>),
    /// Logical OR of two predicates.
    Or(Box<Predicate>, Box<Predicate>),
    /// Always true (useful for optimization passes).
    True,
    /// Always false (useful for optimization passes).
    False,
}

impl Predicate {
    /// Returns all column indices referenced by this predicate.
    ///
    /// This is useful for determining which columns a filter depends on,
    /// which can inform optimizations like filter pushdown.
    ///
    /// # Example
    ///
    /// ```rust
    /// use datalog_ir::Predicate;
    ///
    /// let pred = Predicate::And(
    ///     Box::new(Predicate::ColumnGtConst(0, 5)),
    ///     Box::new(Predicate::ColumnsEq(1, 2)),
    /// );
    ///
    /// let cols = pred.referenced_columns();
    /// assert_eq!(cols.len(), 3);
    /// assert!(cols.contains(&0));
    /// assert!(cols.contains(&1));
    /// assert!(cols.contains(&2));
    /// ```
    pub fn referenced_columns(&self) -> HashSet<usize> {
        let mut cols = HashSet::new();
        self.collect_columns(&mut cols);
        cols
    }

    fn collect_columns(&self, cols: &mut HashSet<usize>) {
        match self {
            Predicate::ColumnEqConst(col, _)
            | Predicate::ColumnNeConst(col, _)
            | Predicate::ColumnGtConst(col, _)
            | Predicate::ColumnLtConst(col, _)
            | Predicate::ColumnGeConst(col, _)
            | Predicate::ColumnLeConst(col, _) => {
                cols.insert(*col);
            }
            Predicate::ColumnsEq(left, right) | Predicate::ColumnsNe(left, right) => {
                cols.insert(*left);
                cols.insert(*right);
            }
            Predicate::And(p1, p2) | Predicate::Or(p1, p2) => {
                p1.collect_columns(cols);
                p2.collect_columns(cols);
            }
            Predicate::True | Predicate::False => {}
        }
    }

    /// Returns `true` if this predicate is [`Predicate::True`].
    #[inline]
    pub fn is_always_true(&self) -> bool {
        matches!(self, Predicate::True)
    }

    /// Returns `true` if this predicate is [`Predicate::False`].
    #[inline]
    pub fn is_always_false(&self) -> bool {
        matches!(self, Predicate::False)
    }

    /// Simplifies the predicate using basic constant folding.
    ///
    /// Performs the following simplifications:
    /// - `True AND x` → `x`
    /// - `x AND True` → `x`
    /// - `False AND x` → `False`
    /// - `True OR x` → `True`
    /// - `False OR x` → `x`
    /// - `x OR False` → `x`
    ///
    /// # Example
    ///
    /// ```rust
    /// use datalog_ir::Predicate;
    ///
    /// let pred = Predicate::And(
    ///     Box::new(Predicate::True),
    ///     Box::new(Predicate::ColumnGtConst(0, 5)),
    /// );
    ///
    /// let simplified = pred.simplify();
    /// assert_eq!(simplified, Predicate::ColumnGtConst(0, 5));
    /// ```
    pub fn simplify(self) -> Self {
        match self {
            Predicate::And(p1, p2) => {
                let p1 = p1.simplify();
                let p2 = p2.simplify();

                if p1.is_always_true() {
                    p2
                } else if p2.is_always_true() {
                    p1
                } else if p1.is_always_false() || p2.is_always_false() {
                    Predicate::False
                } else {
                    Predicate::And(Box::new(p1), Box::new(p2))
                }
            }
            Predicate::Or(p1, p2) => {
                let p1 = p1.simplify();
                let p2 = p2.simplify();

                if p1.is_always_true() || p2.is_always_true() {
                    Predicate::True
                } else if p1.is_always_false() {
                    p2
                } else if p2.is_always_false() {
                    p1
                } else {
                    Predicate::Or(Box::new(p1), Box::new(p2))
                }
            }
            other => other,
        }
    }

    /// Adjusts column indices after a projection.
    ///
    /// When pushing a filter through a `Map` node, the column indices in the
    /// predicate need to be remapped to match the new schema.
    ///
    /// Returns `None` if the predicate references columns that are not present
    /// in the projection.
    ///
    /// # Arguments
    ///
    /// * `projection` - The projection mapping: `projection[new_idx] = old_idx`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use datalog_ir::Predicate;
    ///
    /// // Original predicate on column 1
    /// let pred = Predicate::ColumnGtConst(1, 5);
    ///
    /// // Projection: [1, 0, 2] means new column 0 = old column 1
    /// let projection = vec![1, 0, 2];
    ///
    /// let adjusted = pred.adjust_for_projection(&projection);
    /// // Column 1 is now at position 0
    /// assert_eq!(adjusted, Some(Predicate::ColumnGtConst(0, 5)));
    /// ```
    pub fn adjust_for_projection(&self, projection: &[usize]) -> Option<Self> {
        // Helper: find new index of old column
        let find_new_index =
            |old_idx: usize| -> Option<usize> { projection.iter().position(|&idx| idx == old_idx) };

        match self {
            Predicate::ColumnEqConst(col, val) => {
                find_new_index(*col).map(|new_col| Predicate::ColumnEqConst(new_col, *val))
            }
            Predicate::ColumnNeConst(col, val) => {
                find_new_index(*col).map(|new_col| Predicate::ColumnNeConst(new_col, *val))
            }
            Predicate::ColumnGtConst(col, val) => {
                find_new_index(*col).map(|new_col| Predicate::ColumnGtConst(new_col, *val))
            }
            Predicate::ColumnLtConst(col, val) => {
                find_new_index(*col).map(|new_col| Predicate::ColumnLtConst(new_col, *val))
            }
            Predicate::ColumnGeConst(col, val) => {
                find_new_index(*col).map(|new_col| Predicate::ColumnGeConst(new_col, *val))
            }
            Predicate::ColumnLeConst(col, val) => {
                find_new_index(*col).map(|new_col| Predicate::ColumnLeConst(new_col, *val))
            }
            Predicate::ColumnsEq(left, right) => {
                match (find_new_index(*left), find_new_index(*right)) {
                    (Some(new_left), Some(new_right)) => {
                        Some(Predicate::ColumnsEq(new_left, new_right))
                    }
                    _ => None,
                }
            }
            Predicate::ColumnsNe(left, right) => {
                match (find_new_index(*left), find_new_index(*right)) {
                    (Some(new_left), Some(new_right)) => {
                        Some(Predicate::ColumnsNe(new_left, new_right))
                    }
                    _ => None,
                }
            }
            Predicate::And(p1, p2) => {
                match (
                    p1.adjust_for_projection(projection),
                    p2.adjust_for_projection(projection),
                ) {
                    (Some(new_p1), Some(new_p2)) => {
                        Some(Predicate::And(Box::new(new_p1), Box::new(new_p2)))
                    }
                    (Some(new_p1), None) => Some(new_p1),
                    (None, Some(new_p2)) => Some(new_p2),
                    (None, None) => None,
                }
            }
            Predicate::Or(p1, p2) => {
                match (
                    p1.adjust_for_projection(projection),
                    p2.adjust_for_projection(projection),
                ) {
                    (Some(new_p1), Some(new_p2)) => {
                        Some(Predicate::Or(Box::new(new_p1), Box::new(new_p2)))
                    }
                    _ => None,
                }
            }
            Predicate::True => Some(Predicate::True),
            Predicate::False => Some(Predicate::False),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // IRNode Tests
    // ========================================================================

    #[test]
    fn test_scan_output_schema() {
        let scan = IRNode::Scan {
            relation: "edge".to_string(),
            schema: vec!["x".to_string(), "y".to_string()],
        };
        assert_eq!(scan.output_schema(), vec!["x", "y"]);
        assert!(scan.is_scan());
        assert!(!scan.is_join());
    }

    #[test]
    fn test_filter_passes_through_schema() {
        let scan = IRNode::Scan {
            relation: "edge".to_string(),
            schema: vec!["x".to_string(), "y".to_string()],
        };

        let filter = IRNode::Filter {
            input: Box::new(scan),
            predicate: Predicate::ColumnGtConst(0, 5),
        };

        assert_eq!(filter.output_schema(), vec!["x", "y"]);
    }

    #[test]
    fn test_map_reorders_schema() {
        let scan = IRNode::Scan {
            relation: "edge".to_string(),
            schema: vec!["x".to_string(), "y".to_string()],
        };

        let map = IRNode::Map {
            input: Box::new(scan),
            projection: vec![1, 0],
            output_schema: vec!["y".to_string(), "x".to_string()],
        };

        assert_eq!(map.output_schema(), vec!["y", "x"]);
    }

    #[test]
    fn test_join_output_schema() {
        let left = IRNode::Scan {
            relation: "edge".to_string(),
            schema: vec!["a".to_string(), "b".to_string()],
        };

        let right = IRNode::Scan {
            relation: "node".to_string(),
            schema: vec!["id".to_string(), "label".to_string()],
        };

        let join = IRNode::Join {
            left: Box::new(left),
            right: Box::new(right),
            left_keys: vec![1],
            right_keys: vec![0],
            output_schema: vec![
                "a".to_string(),
                "b".to_string(),
                "id".to_string(),
                "label".to_string(),
            ],
        };

        assert_eq!(join.output_schema(), vec!["a", "b", "id", "label"]);
        assert!(join.is_join());
        assert!(!join.is_scan());
    }

    #[test]
    fn test_distinct_passes_through_schema() {
        let scan = IRNode::Scan {
            relation: "edge".to_string(),
            schema: vec!["x".to_string(), "y".to_string()],
        };

        let distinct = IRNode::Distinct {
            input: Box::new(scan),
        };

        assert_eq!(distinct.output_schema(), vec!["x", "y"]);
    }

    #[test]
    fn test_union_uses_first_input_schema() {
        let scan1 = IRNode::Scan {
            relation: "edge1".to_string(),
            schema: vec!["x".to_string(), "y".to_string()],
        };

        let scan2 = IRNode::Scan {
            relation: "edge2".to_string(),
            schema: vec!["a".to_string(), "b".to_string()],
        };

        let union = IRNode::Union {
            inputs: vec![scan1, scan2],
        };

        assert_eq!(union.output_schema(), vec!["x", "y"]);
    }

    #[test]
    fn test_empty_union_schema() {
        let union = IRNode::Union { inputs: vec![] };
        assert_eq!(union.output_schema(), Vec::<String>::new());
    }

    #[test]
    fn test_pretty_print_contains_operators() {
        let plan = IRNode::Filter {
            input: Box::new(IRNode::Scan {
                relation: "edge".to_string(),
                schema: vec!["x".to_string(), "y".to_string()],
            }),
            predicate: Predicate::ColumnGtConst(0, 5),
        };

        let output = plan.pretty_print(0);
        assert!(output.contains("Filter"));
        assert!(output.contains("Scan"));
        assert!(output.contains("edge"));
    }

    // ========================================================================
    // Predicate Tests
    // ========================================================================

    #[test]
    fn test_predicate_referenced_columns_simple() {
        let pred = Predicate::ColumnEqConst(2, 42);
        let cols = pred.referenced_columns();
        assert_eq!(cols.len(), 1);
        assert!(cols.contains(&2));
    }

    #[test]
    fn test_predicate_referenced_columns_compound() {
        let pred = Predicate::And(
            Box::new(Predicate::ColumnGtConst(0, 5)),
            Box::new(Predicate::ColumnsEq(1, 2)),
        );

        let cols = pred.referenced_columns();
        assert_eq!(cols.len(), 3);
        assert!(cols.contains(&0));
        assert!(cols.contains(&1));
        assert!(cols.contains(&2));
    }

    #[test]
    fn test_predicate_referenced_columns_or() {
        let pred = Predicate::Or(
            Box::new(Predicate::ColumnLtConst(3, 10)),
            Box::new(Predicate::ColumnsNe(4, 5)),
        );

        let cols = pred.referenced_columns();
        assert_eq!(cols.len(), 3);
        assert!(cols.contains(&3));
        assert!(cols.contains(&4));
        assert!(cols.contains(&5));
    }

    #[test]
    fn test_predicate_referenced_columns_true_false() {
        assert!(Predicate::True.referenced_columns().is_empty());
        assert!(Predicate::False.referenced_columns().is_empty());
    }

    #[test]
    fn test_predicate_is_always_true_false() {
        assert!(Predicate::True.is_always_true());
        assert!(!Predicate::True.is_always_false());
        assert!(Predicate::False.is_always_false());
        assert!(!Predicate::False.is_always_true());

        let pred = Predicate::ColumnEqConst(0, 1);
        assert!(!pred.is_always_true());
        assert!(!pred.is_always_false());
    }

    #[test]
    fn test_predicate_simplify_and_true() {
        let pred = Predicate::And(
            Box::new(Predicate::True),
            Box::new(Predicate::ColumnGtConst(0, 5)),
        );
        assert_eq!(pred.simplify(), Predicate::ColumnGtConst(0, 5));

        let pred2 = Predicate::And(
            Box::new(Predicate::ColumnGtConst(0, 5)),
            Box::new(Predicate::True),
        );
        assert_eq!(pred2.simplify(), Predicate::ColumnGtConst(0, 5));
    }

    #[test]
    fn test_predicate_simplify_and_false() {
        let pred = Predicate::And(
            Box::new(Predicate::False),
            Box::new(Predicate::ColumnGtConst(0, 5)),
        );
        assert_eq!(pred.simplify(), Predicate::False);

        let pred2 = Predicate::And(
            Box::new(Predicate::ColumnGtConst(0, 5)),
            Box::new(Predicate::False),
        );
        assert_eq!(pred2.simplify(), Predicate::False);
    }

    #[test]
    fn test_predicate_simplify_or_true() {
        let pred = Predicate::Or(
            Box::new(Predicate::True),
            Box::new(Predicate::ColumnGtConst(0, 5)),
        );
        assert_eq!(pred.simplify(), Predicate::True);
    }

    #[test]
    fn test_predicate_simplify_or_false() {
        let pred = Predicate::Or(
            Box::new(Predicate::False),
            Box::new(Predicate::ColumnGtConst(0, 5)),
        );
        assert_eq!(pred.simplify(), Predicate::ColumnGtConst(0, 5));

        let pred2 = Predicate::Or(
            Box::new(Predicate::ColumnGtConst(0, 5)),
            Box::new(Predicate::False),
        );
        assert_eq!(pred2.simplify(), Predicate::ColumnGtConst(0, 5));
    }

    #[test]
    fn test_predicate_simplify_nested() {
        let pred = Predicate::And(
            Box::new(Predicate::Or(
                Box::new(Predicate::False),
                Box::new(Predicate::ColumnEqConst(0, 1)),
            )),
            Box::new(Predicate::True),
        );
        assert_eq!(pred.simplify(), Predicate::ColumnEqConst(0, 1));
    }

    #[test]
    fn test_predicate_adjust_for_projection_simple() {
        let projection = vec![1, 0, 2];
        let pred = Predicate::ColumnGtConst(1, 5);

        let adjusted = pred.adjust_for_projection(&projection);
        assert_eq!(adjusted, Some(Predicate::ColumnGtConst(0, 5)));
    }

    #[test]
    fn test_predicate_adjust_for_projection_missing_column() {
        let projection = vec![0, 2];
        let pred = Predicate::ColumnGtConst(1, 5);

        let adjusted = pred.adjust_for_projection(&projection);
        assert_eq!(adjusted, None);
    }

    #[test]
    fn test_predicate_adjust_for_projection_columns_eq() {
        let projection = vec![2, 0, 1];
        let pred = Predicate::ColumnsEq(0, 1);

        let adjusted = pred.adjust_for_projection(&projection);
        assert_eq!(adjusted, Some(Predicate::ColumnsEq(1, 2)));
    }

    #[test]
    fn test_predicate_adjust_for_projection_and() {
        let projection = vec![1, 0];
        let pred = Predicate::And(
            Box::new(Predicate::ColumnEqConst(0, 5)),
            Box::new(Predicate::ColumnGtConst(1, 10)),
        );

        let adjusted = pred.adjust_for_projection(&projection);
        assert_eq!(
            adjusted,
            Some(Predicate::And(
                Box::new(Predicate::ColumnEqConst(1, 5)),
                Box::new(Predicate::ColumnGtConst(0, 10)),
            ))
        );
    }

    #[test]
    fn test_predicate_adjust_for_projection_and_partial() {
        let projection = vec![0];
        let pred = Predicate::And(
            Box::new(Predicate::ColumnEqConst(0, 5)),
            Box::new(Predicate::ColumnGtConst(1, 10)),
        );

        let adjusted = pred.adjust_for_projection(&projection);
        assert_eq!(adjusted, Some(Predicate::ColumnEqConst(0, 5)));
    }

    #[test]
    fn test_predicate_adjust_for_projection_or_requires_both() {
        let projection = vec![0];
        let pred = Predicate::Or(
            Box::new(Predicate::ColumnEqConst(0, 5)),
            Box::new(Predicate::ColumnGtConst(1, 10)),
        );

        let adjusted = pred.adjust_for_projection(&projection);
        assert_eq!(adjusted, None);
    }

    #[test]
    fn test_predicate_adjust_true_false() {
        let projection = vec![0];
        assert_eq!(
            Predicate::True.adjust_for_projection(&projection),
            Some(Predicate::True)
        );
        assert_eq!(
            Predicate::False.adjust_for_projection(&projection),
            Some(Predicate::False)
        );
    }

    #[test]
    fn test_predicate_all_comparison_types() {
        let pred_eq = Predicate::ColumnEqConst(0, 1);
        let pred_ne = Predicate::ColumnNeConst(0, 1);
        let pred_gt = Predicate::ColumnGtConst(0, 1);
        let pred_lt = Predicate::ColumnLtConst(0, 1);
        let pred_ge = Predicate::ColumnGeConst(0, 1);
        let pred_le = Predicate::ColumnLeConst(0, 1);

        for pred in [pred_eq, pred_ne, pred_gt, pred_lt, pred_ge, pred_le] {
            let cols = pred.referenced_columns();
            assert_eq!(cols.len(), 1);
            assert!(cols.contains(&0));
        }
    }

    #[test]
    fn test_irnode_clone_and_eq() {
        let scan = IRNode::Scan {
            relation: "test".to_string(),
            schema: vec!["a".to_string()],
        };

        let scan_clone = scan.clone();
        assert_eq!(scan, scan_clone);
    }

    #[test]
    fn test_predicate_clone_and_eq() {
        let pred = Predicate::And(
            Box::new(Predicate::ColumnEqConst(0, 1)),
            Box::new(Predicate::True),
        );

        let pred_clone = pred.clone();
        assert_eq!(pred, pred_clone);
    }
}
