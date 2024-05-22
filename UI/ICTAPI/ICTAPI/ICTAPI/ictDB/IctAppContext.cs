using System;
using System.Collections.Generic;
using Microsoft.EntityFrameworkCore;

namespace ICTAPI.ictDB;

public partial class IctAppContext : DbContext
{
    public IctAppContext()
    {
    }

    public IctAppContext(DbContextOptions<IctAppContext> options)
        : base(options)
    {
    }

    public virtual DbSet<User> Users { get; set; }

    public virtual DbSet<WorkContent> WorkContents { get; set; }

    //    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    //#warning To protect potentially sensitive information in your connection string, you should move it out of source code. You can avoid scaffolding the connection string by using the Name= syntax to read it from configuration - see https://go.microsoft.com/fwlink/?linkid=2131148. For more guidance on storing connection strings, see http://go.microsoft.com/fwlink/?LinkId=723263.
    //        => optionsBuilder.UseMySQL("server=74.208.107.56;uid=ictApp;pwd=ictApp123;database=ictApp");

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<User>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PRIMARY");

            entity.ToTable("users");

            entity.Property(e => e.Id)
                .HasColumnType("int(11)")
                .HasColumnName("id");
            entity.Property(e => e.CreatedAt)
                .HasMaxLength(255)
                .HasDefaultValueSql("'NULL'")
                .HasColumnName("created_at");
            entity.Property(e => e.EditedAt)
                .HasMaxLength(255)
                .HasDefaultValueSql("'NULL'")
                .HasColumnName("edited_at");
            entity.Property(e => e.Email)
                .HasMaxLength(255)
                .HasColumnName("email");
            entity.Property(e => e.FName)
                .HasMaxLength(255)
                .HasColumnName("f_name");
            entity.Property(e => e.LName)
                .HasMaxLength(255)
                .HasColumnName("l_name");
            entity.Property(e => e.Password)
                .HasMaxLength(255)
                .HasColumnName("password");
            entity.Property(e => e.Token)
                .HasMaxLength(255)
                .HasDefaultValueSql("'NULL'")
                .HasColumnName("token");
        });

        modelBuilder.Entity<WorkContent>(entity =>
        {
            entity.HasKey(e => e.Id).HasName("PRIMARY");

            entity.ToTable("workContent");

            entity.HasIndex(e => e.UserId, "user_id");

            entity.Property(e => e.Id)
                .HasColumnType("int(11)")
                .HasColumnName("id");
            entity.Property(e => e.CreatedAt)
                .HasDefaultValueSql("'NULL'")
                .HasColumnType("datetime")
                .HasColumnName("created_at");
            entity.Property(e => e.FileName)
                .HasMaxLength(255)
                .HasColumnName("file_name");
            entity.Property(e => e.FinishedAt)
                .HasDefaultValueSql("'NULL'")
                .HasColumnType("datetime")
                .HasColumnName("finished_at");
            entity.Property(e => e.InputPath)
                .HasMaxLength(255)
                .HasColumnName("input_path");
            entity.Property(e => e.OutputPath)
                .HasMaxLength(255)
                .HasDefaultValueSql("'NULL'")
                .HasColumnName("output_path");
            entity.Property(e => e.UserId)
                .HasColumnType("int(11)")
                .HasColumnName("user_id");
            //entity.Property(e => e.PatientId)
            //    .HasColumnType("int(11)")
            //    .HasColumnName("patient_id");
            entity.HasOne(d => d.User).WithMany(p => p.WorkContents)
                .HasForeignKey(d => d.UserId)
                .OnDelete(DeleteBehavior.Restrict)
                .HasConstraintName("workContent_ibfk_1");
        });

        OnModelCreatingPartial(modelBuilder);
    }

    partial void OnModelCreatingPartial(ModelBuilder modelBuilder);
}
