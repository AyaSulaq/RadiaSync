using System;
using System.Collections.Generic;

namespace ICTAPI.ictDB;

public partial class WorkContent
{
    public int Id { get; set; }

    public int UserId { get; set; }

    public string InputPath { get; set; } = null!;

    public string? OutputPath { get; set; }

    public string FileName { get; set; } = null!;

    public DateTime? CreatedAt { get; set; }

    public DateTime? FinishedAt { get; set; }

    public virtual User User { get; set; } = null!;

    //  public StreamContent StreamFile { get; set; }

   // public int PatientId { get; set; }
}
