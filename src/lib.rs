pub use arrow_array;
pub use arrow_schema;

pub use rusqlite;

use std::io;
use std::str::Utf8Error;
use std::sync::Arc;

use arrow_schema::SchemaRef;

use arrow_array::ArrayRef;
use arrow_array::RecordBatch;

use arrow_array::builder::ArrayBuilder;
use arrow_array::builder::BinaryBuilder;
use arrow_array::builder::Float64Builder;
use arrow_array::builder::Int64Builder;
use arrow_array::builder::StringBuilder;

use rusqlite::Row;
use rusqlite::Rows;

use rusqlite::types::ValueRef;

pub trait ArrowColumnVisitor: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn visit_null(&mut self) -> Result<(), Self::Error>;
    fn visit_integer(&mut self, val: i64) -> Result<(), Self::Error>;
    fn visit_real(&mut self, val: f64) -> Result<(), Self::Error>;
    fn visit_text(&mut self, val: &str) -> Result<(), Self::Error>;
    fn visit_blob(&mut self, val: &[u8]) -> Result<(), Self::Error>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        0 == self.len()
    }

    fn finish(&mut self) -> ArrayRef;

    fn lossy_conv_or_error(&self, _rejected: &[u8]) -> Option<Self::Error> {
        None
    }

    fn accept_cell(&mut self, val: ValueRef) -> Result<(), Self::Error> {
        match val {
            ValueRef::Null => self.visit_null(),
            ValueRef::Integer(v) => self.visit_integer(v),
            ValueRef::Real(v) => self.visit_real(v),
            ValueRef::Text(v) => {
                let rslt: Result<&str, Utf8Error> = std::str::from_utf8(v);
                if let Ok(s) = rslt {
                    return self.visit_text(s);
                };

                let oerr: Option<Self::Error> = self.lossy_conv_or_error(v);
                if let Some(err) = oerr {
                    return Err(err);
                };

                let lossy: &str = &String::from_utf8_lossy(v);
                self.visit_text(lossy)
            }
            ValueRef::Blob(v) => self.visit_blob(v),
        }
    }
}

pub enum Builder {
    Integer(Int64Builder),
    Real(Float64Builder),
    Text(StringBuilder),
    Blob(BinaryBuilder),
    Custom(Box<dyn ArrowColumnVisitor<Error = io::Error>>),
}

impl Builder {
    pub fn append(&mut self, val: ValueRef) -> Result<(), io::Error> {
        match (self, val) {
            (Self::Integer(b), ValueRef::Integer(v)) => b.append_value(v),
            (Self::Real(b), ValueRef::Real(v)) => b.append_value(v),
            (Self::Text(b), ValueRef::Text(v)) => {
                let s: &str = std::str::from_utf8(v).unwrap_or_default();
                b.append_value(s)
            }
            (Self::Blob(b), ValueRef::Blob(v)) => b.append_value(v),

            (Self::Integer(b), ValueRef::Null) => b.append_null(),
            (Self::Real(b), ValueRef::Null) => b.append_null(),
            (Self::Text(b), ValueRef::Null) => b.append_null(),
            (Self::Blob(b), ValueRef::Null) => b.append_null(),

            (Self::Custom(b), v) => b.accept_cell(v)?,

            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "value type unmatch",
                ));
            }
        }
        Ok(())
    }

    pub fn finish(&mut self) -> ArrayRef {
        match self {
            Self::Integer(b) => Arc::new(b.finish()),
            Self::Real(b) => Arc::new(b.finish()),
            Self::Text(b) => Arc::new(b.finish()),
            Self::Blob(b) => Arc::new(b.finish()),
            Self::Custom(b) => Arc::new(b.finish()),
        }
    }
}

impl Builder {
    pub fn len(&self) -> usize {
        match self {
            Self::Integer(b) => b.len(),
            Self::Real(b) => b.len(),
            Self::Text(b) => b.len(),
            Self::Blob(b) => b.len(),
            Self::Custom(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        0 == self.len()
    }
}

impl Builder {
    pub fn to_batch(bldrs: &mut [Self], sch: SchemaRef) -> Result<RecordBatch, io::Error> {
        let iref = bldrs.iter_mut().map(|b| b.finish());
        let v: Vec<ArrayRef> = iref.collect();
        RecordBatch::try_new(sch, v).map_err(io::Error::other)
    }
}

pub const BATCH_SIZE_DEFAULT: usize = 8192;

pub fn rows2rbat(
    rows: &mut Rows,
    bat_size: usize,
    builders: &mut [Builder],
    sch: SchemaRef,
) -> Result<RecordBatch, io::Error> {
    if bat_size < 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "batch size minimal size: 1",
        ));
    }

    loop {
        let ob1st: Option<&Builder> = builders.iter().next();
        let Some(b1st) = ob1st else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "empty builders",
            ));
        };

        let sz: usize = b1st.len();
        if bat_size <= sz {
            return Builder::to_batch(builders, sch);
        }

        let orow: Option<&Row> = rows.next().map_err(io::Error::other)?;
        let Some(row) = orow else {
            if sz < 1 {
                return Ok(RecordBatch::new_empty(sch));
            }

            return Builder::to_batch(builders, sch);
        };

        for pair in builders.iter_mut().enumerate() {
            let (ix, bldr) = pair;
            let vref: ValueRef = row.get_ref(ix).map_err(io::Error::other)?;
            bldr.append(vref)?;
        }
    }
}

pub fn rows2rbat_default(
    rows: &mut Rows,
    builders: &mut [Builder],
    sch: SchemaRef,
) -> Result<RecordBatch, io::Error> {
    rows2rbat(rows, BATCH_SIZE_DEFAULT, builders, sch)
}

#[cfg(test)]
mod tests {
    mod rows2rbat {
        use arrow_schema::Schema;
        use arrow_schema::SchemaRef;

        use arrow_schema::DataType;
        use arrow_schema::Field;

        use arrow_array::RecordBatch;

        use arrow_array::builder::Int64Builder;
        use arrow_array::builder::StringBuilder;

        use arrow_array::cast::AsArray;

        use arrow_array::Array;
        use arrow_array::ArrayRef;

        use arrow_array::BinaryArray;
        use arrow_array::Float64Array;
        use arrow_array::StringArray;

        use rusqlite::Connection;
        use rusqlite::Rows;
        use rusqlite::Statement;

        use crate::Builder;

        fn simple_sch() -> SchemaRef {
            Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("severity", DataType::Utf8, false),
                Field::new("body", DataType::Utf8, false),
            ])
            .into()
        }

        fn simple_sch_null() -> SchemaRef {
            Schema::new(vec![
                Field::new("timestamp", DataType::Int64, true),
                Field::new("severity", DataType::Utf8, true),
                Field::new("body", DataType::Utf8, true),
            ])
            .into()
        }

        fn simple_builders() -> Vec<Builder> {
            vec![
                Builder::Integer(Int64Builder::default()),
                Builder::Text(StringBuilder::default()),
                Builder::Text(StringBuilder::default()),
            ]
        }

        fn empty_connection() -> Connection {
            Connection::open_in_memory().unwrap()
        }

        fn simple_connection() -> Connection {
            let c: Connection = empty_connection();
            c.execute(
                r#"
                    CREATE TABLE IF NOT EXISTS log(
                        id INTEGER PRIMARY KEY,
                        timestamp INTEGER NOT NULL,
                        severity TEXT NOT NULL,
                        body TEXT NOT NULL
                    )
                "#,
                (),
            )
            .unwrap();
            c
        }

        const SIMPLE_SEL: &str = r#"
            SELECT
                timestamp,
                severity,
                body
            FROM log
            ORDER BY rowid
        "#;

        #[test]
        fn empty() {
            let lc: Connection = simple_connection();
            let mut stmt: Statement = lc.prepare(SIMPLE_SEL).unwrap();
            let mut rows: Rows = stmt.query(()).unwrap();

            let mut bldrs: Vec<Builder> = simple_builders();

            let rbat: RecordBatch =
                crate::rows2rbat(&mut rows, 1024, &mut bldrs, simple_sch()).unwrap();

            assert_eq!(0, rbat.num_rows());
        }

        #[test]
        fn single() {
            let lc: Connection = simple_connection();

            lc.execute(
                r#"
                    INSERT INTO log(timestamp, severity, body)
                    VALUES(?1,?2,?3)
                "#,
                (299792458, "INFO", "apt update done"),
            )
            .unwrap();

            let mut stmt: Statement = lc.prepare(SIMPLE_SEL).unwrap();
            let mut rows: Rows = stmt.query(()).unwrap();

            let mut bldrs: Vec<Builder> = simple_builders();

            let rbat: RecordBatch =
                crate::rows2rbat(&mut rows, 1024, &mut bldrs, simple_sch()).unwrap();

            assert_eq!(1, rbat.num_rows());
        }

        #[test]
        fn unit_batch() {
            let lc: Connection = simple_connection();

            lc.execute(
                r#"
                    INSERT INTO log(timestamp, severity, body)
                    VALUES(?1,?2,?3)
                "#,
                (634, "INFO", "apt update done"),
            )
            .unwrap();

            lc.execute(
                r#"
                    INSERT INTO log(timestamp, severity, body)
                    VALUES(?1,?2,?3)
                "#,
                (299792458, "INFO", "apt update done"),
            )
            .unwrap();

            let mut stmt: Statement = lc.prepare(SIMPLE_SEL).unwrap();
            let mut rows: Rows = stmt.query(()).unwrap();

            let mut bldrs: Vec<Builder> = simple_builders();

            let rbat: RecordBatch =
                crate::rows2rbat(&mut rows, 1, &mut bldrs, simple_sch()).unwrap();

            assert_eq!(1, rbat.num_rows());
        }

        #[test]
        fn tiny_batch() {
            let lc: Connection = simple_connection();

            lc.execute(
                r#"
                    INSERT INTO log(timestamp, severity, body)
                    VALUES(?1,?2,?3)
                "#,
                (599, "INFO", "apt update done"),
            )
            .unwrap();

            lc.execute(
                r#"
                    INSERT INTO log(timestamp, severity, body)
                    VALUES(?1,?2,?3)
                "#,
                (634, "INFO", "apt update done"),
            )
            .unwrap();

            lc.execute(
                r#"
                    INSERT INTO log(timestamp, severity, body)
                    VALUES(?1,?2,?3)
                "#,
                (299792458, "INFO", "apt update done"),
            )
            .unwrap();

            let mut stmt: Statement = lc.prepare(SIMPLE_SEL).unwrap();
            let mut rows: Rows = stmt.query(()).unwrap();

            let mut bldrs: Vec<Builder> = simple_builders();

            let rbat: RecordBatch =
                crate::rows2rbat(&mut rows, 2, &mut bldrs, simple_sch()).unwrap();

            assert_eq!(2, rbat.num_rows());

            let rbat: RecordBatch =
                crate::rows2rbat(&mut rows, 2, &mut bldrs, simple_sch()).unwrap();

            assert_eq!(1, rbat.num_rows());
        }

        #[test]
        fn null_values() {
            let lc: Connection = empty_connection();
            lc.execute(
                r#"
                    CREATE TABLE log(
                        timestamp INTEGER, 
                        severity TEXT, 
                        body TEXT
                    )
                "#,
                (),
            )
            .unwrap();

            lc.execute("INSERT INTO log VALUES (1, 'INFO', 'ok')", ())
                .unwrap();
            lc.execute("INSERT INTO log VALUES (2, NULL, NULL)", ())
                .unwrap();

            let mut stmt: Statement = lc.prepare(SIMPLE_SEL).unwrap();
            let mut rows: Rows = stmt.query(()).unwrap();
            let mut bldrs: Vec<Builder> = simple_builders();

            let rbat: RecordBatch =
                crate::rows2rbat(&mut rows, 10, &mut bldrs, simple_sch_null()).unwrap();

            assert_eq!(2, rbat.num_rows());

            let severity_col: &ArrayRef = rbat.column(1);
            let strings: &StringArray = severity_col.as_string();
            assert!(!strings.is_null(0));
            assert!(strings.is_null(1));
        }

        #[test]
        fn real_values() {
            let lc: Connection = empty_connection();
            lc.execute(r#"CREATE TABLE floats(val REAL)"#, ()).unwrap();
            lc.execute("INSERT INTO floats VALUES (1.23)", ()).unwrap();
            lc.execute("INSERT INTO floats VALUES (4.56)", ()).unwrap();

            let mut stmt: Statement = lc.prepare("SELECT val FROM floats").unwrap();
            let mut rows: Rows = stmt.query(()).unwrap();

            let mut bldrs: Vec<Builder> = vec![Builder::Real(
                arrow_array::builder::Float64Builder::default(),
            )];
            let sch = Schema::new(vec![Field::new("val", DataType::Float64, false)]).into();

            let rbat: RecordBatch = crate::rows2rbat(&mut rows, 10, &mut bldrs, sch).unwrap();

            assert_eq!(2, rbat.num_rows());
            let col: &Float64Array = rbat.column(0).as_primitive();
            assert_eq!(1.23, col.value(0));
            assert_eq!(4.56, col.value(1));
        }

        #[test]
        fn blob_values() {
            let lc: Connection = empty_connection();
            lc.execute(r#"CREATE TABLE blobs(data BLOB)"#, ()).unwrap();
            lc.execute("INSERT INTO blobs VALUES (?)", (vec![1, 2, 3],))
                .unwrap();
            lc.execute("INSERT INTO blobs VALUES (?)", (vec![4, 5, 6],))
                .unwrap();

            let mut stmt: Statement = lc.prepare("SELECT data FROM blobs").unwrap();
            let mut rows: Rows = stmt.query(()).unwrap();

            let mut bldrs: Vec<Builder> =
                vec![Builder::Blob(arrow_array::builder::BinaryBuilder::default())];
            let sch = Schema::new(vec![Field::new("data", DataType::Binary, false)]).into();

            let rbat: RecordBatch = crate::rows2rbat(&mut rows, 10, &mut bldrs, sch).unwrap();

            assert_eq!(2, rbat.num_rows());
            let col: &BinaryArray = rbat.column(0).as_binary();
            assert_eq!(vec![1, 2, 3], col.value(0));
            assert_eq!(vec![4, 5, 6], col.value(1));
        }

        #[test]
        fn type_mismatch_error() {
            let lc: Connection = empty_connection();
            lc.execute("CREATE TABLE mismatch(val TEXT)", ()).unwrap();
            lc.execute("INSERT INTO mismatch VALUES ('not a number')", ())
                .unwrap();

            let mut stmt: Statement = lc.prepare("SELECT val FROM mismatch").unwrap();
            let mut rows: Rows = stmt.query(()).unwrap();

            // Using Integer builder for a Text column
            let mut bldrs: Vec<Builder> = vec![Builder::Integer(
                arrow_array::builder::Int64Builder::default(),
            )];
            let sch = Schema::new(vec![Field::new("val", DataType::Int64, false)]).into();

            let result = crate::rows2rbat(&mut rows, 10, &mut bldrs, sch);

            assert!(result.is_err());
            assert_eq!(result.unwrap_err().to_string(), "value type unmatch");
        }
    }
}
